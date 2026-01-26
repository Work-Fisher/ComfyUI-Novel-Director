import json
import torch
import re
import torch.nn.functional as F

class ScriptJSONParser:
    """
    节点1：剧本JSON解析器 (指令前置优化版)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "JSON剧本数据": ("STRING", {"multiline": True, "forceInput": True}), 
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING") 
    RETURN_NAMES = ("人设提示词列表", "基础分镜提示词", "主角索引列表(A)", "配角索引列表(B)", "合成后的中文提示词")
    OUTPUT_IS_LIST = (True, True, True, True, True) 
    FUNCTION = "parse_script"
    CATEGORY = "Novel Director"

    def parse_script(self, JSON剧本数据):
        char_prompts = []
        char_names = [] 
        scene_prompts = []
        ref_indices_a = []
        ref_indices_b = []
        merged_prompts = [] 

        print(f"\n🔵 [Novel Director] 开始解析剧本...")

        try:
            # --- 1. 数据清洗 ---
            raw_text = JSON剧本数据
            if isinstance(raw_text, list): raw_text = raw_text[0]
            
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            clean_text = match.group(0) if match else raw_text
            clean_text = clean_text.replace("```json", "").replace("```", "")
            
            try: 
                data = json.loads(clean_text)
            except: 
                print("JSON解析失败，尝试宽松模式...")
                data = {}

            # --- 2. 提取人设 ---
            c_list = data.get("character_ref_prompts") or data.get("character_list") or []
            for idx, item in enumerate(c_list):
                prompt = item.get("prompt", "") if isinstance(item, dict) else item
                name = item.get("name", f"Char_{idx}") if isinstance(item, dict) else item[:10]
                char_prompts.append(prompt)
                char_names.append(name)
                print(f"  👤 角色[{idx}]: {name}")

            # --- 3. 提取分镜 ---
            s_list = data.get("storyboard_list") or data.get("storyboard") or []
            
            for s_idx, item in enumerate(s_list):
                if isinstance(item, dict):
                    base_prompt = item.get("prompt", "")
                    scene_prompts.append(base_prompt)
                    
                    target_obj = item.get("main_character", "")
                    explicit_idx = item.get("ref_image_index", None)
                    
                    found_indices = []
                    if explicit_idx is not None:
                        if isinstance(explicit_idx, list): found_indices = [int(x) for x in explicit_idx]
                        else: 
                            try: found_indices = [int(explicit_idx)]
                            except: pass
                    
                    if not found_indices and target_obj:
                        targets_to_check = target_obj if isinstance(target_obj, list) else [target_obj]
                        detected = []
                        for t_str in targets_to_check:
                            t_str = str(t_str).lower().strip()
                            if t_str == "none": continue
                            for idx, registered_name in enumerate(char_names):
                                reg_clean = registered_name.lower().strip()
                                if reg_clean and (reg_clean in t_str or t_str in reg_clean):
                                    if idx not in detected: detected.append(idx)
                        found_indices = detected

                    idx_a = found_indices[0] if len(found_indices) > 0 else -1
                    idx_b = found_indices[1] if len(found_indices) > 1 else -1
                    
                    ref_indices_a.append(idx_a)
                    ref_indices_b.append(idx_b)

                    # --- 4. 生成前置指令 ---
                    instruction = ""
                    if idx_a != -1 and idx_b != -1:
                        name_a = char_names[idx_a]
                        name_b = char_names[idx_b]
                        instruction = f"画面包含两人，请严格参考图一的角色【{name_a}】和图二的角色【{name_b}】，保持人物特征一致"
                    elif idx_a != -1:
                        name_a = char_names[idx_a]
                        instruction = f"画面包含一人，请严格参考图一的角色【{name_a}】，保持人物特征一致"
                    elif idx_b != -1:
                        name_b = char_names[idx_b]
                        instruction = f"画面包含一人，请严格参考图二的角色【{name_b}】，保持人物特征一致"
                    else:
                        instruction = "高质量场景，无人"

                    final_prompt = f"{instruction}。{base_prompt}"
                    merged_prompts.append(final_prompt)
                    
                    print(f"  🎬 分镜[{s_idx+1}] 提示词生成完毕")

            # --- 5. 补齐 ---
            target_len = len(scene_prompts)
            if len(ref_indices_a) < target_len: ref_indices_a.extend([-1] * (target_len - len(ref_indices_a)))
            if len(ref_indices_b) < target_len: ref_indices_b.extend([-1] * (target_len - len(ref_indices_b)))

            return (char_prompts, scene_prompts, ref_indices_a, ref_indices_b, merged_prompts)

        except Exception as e:
            print(f"🔴 解析严重错误: {e}")
            return (["Error"], ["Error"], [-1], [-1], ["Error"])


class BatchImageSelector:
    """
    节点2：FLUX专用图片提取器 (修复报错版)
    修复了输入为空时导致 'NoneType object has no attribute shape' 的问题。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "所有角色图Batch": ("IMAGE", ), 
                "索引列表": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("VAE参考图", )
    INPUT_IS_LIST = True 
    OUTPUT_IS_LIST = (True,) 
    FUNCTION = "select_images"
    CATEGORY = "Novel Director"

    def select_images(self, 所有角色图Batch, 索引列表):
        output_images = []
        raw_input = 所有角色图Batch
        image_pool = []
        
        # --- 1. 安全整理图片池 (防爆逻辑) ---
        if raw_input is None:
            print("⚠️ [BatchImageSelector] 警告: 未检测到输入图片!")
        elif isinstance(raw_input, list):
            for i, item in enumerate(raw_input):
                if item is None: continue # 跳过空数据
                try:
                    if hasattr(item, "shape") and len(item.shape) == 4: 
                        for j in range(item.shape[0]): image_pool.append(item[j])
                    elif hasattr(item, "shape"):
                        image_pool.append(item)
                except Exception as e:
                    print(f"⚠️ 读取第 {i} 张图时出错: {e}")
        else:
            if hasattr(raw_input, "shape"):
                for i in range(raw_input.shape[0]): image_pool.append(raw_input[i])

        print(f"🔵 [BatchImageSelector] 成功加载 {len(image_pool)} 张角色参考图")

        # --- 2. 确定基准尺寸 ---
        target_h, target_w = 1024, 1024
        if len(image_pool) > 0:
            target_h, target_w = image_pool[0].shape[0], image_pool[0].shape[1]

        # --- 3. 提取并处理 ---
        for idx in 索引列表:
            try: idx = int(idx)
            except: idx = -1
            
            target_img = None
            
            # 如果索引无效，或者图池是空的 -> 给全黑图
            if idx < 0 or idx >= len(image_pool):
                target_img = torch.zeros((1, target_h, target_w, 3), dtype=torch.float32)
            else:
                img = image_pool[idx]
                # 再次检查取出来的图是否有效
                if img is None:
                     target_img = torch.zeros((1, target_h, target_w, 3), dtype=torch.float32)
                else:
                    # 尺寸对齐
                    try:
                        if img.shape[0] != target_h or img.shape[1] != target_w:
                            img_permuted = img.permute(2, 0, 1).unsqueeze(0)
                            img_resized = F.interpolate(img_permuted, size=(target_h, target_w), mode="bilinear", align_corners=False)
                            target_img = img_resized.permute(0, 2, 3, 1)
                        else:
                            target_img = img.unsqueeze(0)
                    except:
                        # 万一缩放报错，给黑图兜底
                        target_img = torch.zeros((1, target_h, target_w, 3), dtype=torch.float32)
            
            output_images.append(target_img)

        return (output_images, )


class DynamicCharMask:
    """
    节点3：存在遮罩生成器
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "参考图Batch": ("IMAGE", ), 
                "索引列表": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("VAE遮罩", )
    INPUT_IS_LIST = True 
    OUTPUT_IS_LIST = (True,) 
    FUNCTION = "generate_mask"
    CATEGORY = "Novel Director"

    def generate_mask(self, 参考图Batch, 索引列表):
        output_masks = []
        
        # 安全获取参考尺寸
        H, W = 1024, 1024
        try:
            ref_img_list = 参考图Batch[0] if isinstance(参考图Batch, list) else 参考图Batch
            if isinstance(ref_img_list, list) and len(ref_img_list) > 0: 
                ref_img = ref_img_list[0]
            else: 
                ref_img = ref_img_list
            
            if ref_img is not None and hasattr(ref_img, "shape"):
                if len(ref_img.shape) == 4: _, H, W, _ = ref_img.shape
                else: H, W, _ = ref_img.shape
        except:
            pass
            
        device = torch.device("cpu") # 默认

        for idx in 索引列表:
            idx = int(idx)
            if idx == -1:
                mask = torch.zeros((H, W), dtype=torch.float32, device=device)
            else:
                mask = torch.ones((H, W), dtype=torch.float32, device=device)
            output_masks.append(mask)

        return (output_masks, )

NODE_CLASS_MAPPINGS = { 
    "ScriptJSONParser": ScriptJSONParser, 
    "BatchImageSelector": BatchImageSelector,
    "DynamicCharMask": DynamicCharMask
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "ScriptJSONParser": "剧本JSON解析器 (Novel)", 
    "BatchImageSelector": "按索引提取参考图 (FLUX VAE版)",
    "DynamicCharMask": "生成角色存在遮罩 (Novel)"
}
