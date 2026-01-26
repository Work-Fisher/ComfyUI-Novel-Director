import json
import torch

class ScriptJSONParser:
    """
    节点1：剧本JSON解析器 (双人智能版)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "JSON剧本数据": ("STRING", {"multiline": True, "forceInput": True}), 
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT") 
    RETURN_NAMES = ("人设提示词列表", "分镜提示词列表", "主角索引列表(A)", "配角索引列表(B)")
    OUTPUT_IS_LIST = (True, True, True, True) 
    FUNCTION = "parse_script"
    CATEGORY = "Novel Director"

    def parse_script(self, JSON剧本数据):
        char_prompts = []
        char_names = [] 
        scene_prompts = []
        ref_indices_a = []
        ref_indices_b = []

        try:
            # --- 1. 数据清洗 ---
            raw_text = JSON剧本数据
            if isinstance(raw_text, list): raw_text = raw_text[0]
            clean_text = raw_text.strip().replace("```json", "").replace("```", "")
            
            try: data = json.loads(clean_text)
            except: 
                try: data = json.loads(clean_text + "}")
                except: data = {}
            
            # --- 2. 提取人设 ---
            c_list = data.get("character_ref_prompts") or data.get("character_list") or []
            for item in c_list:
                prompt = ""
                name = ""
                if isinstance(item, dict):
                    prompt = item.get("prompt", "")
                    name = item.get("name", "未命名")
                elif isinstance(item, str):
                    prompt = item
                    name = item[:10]
                char_prompts.append(prompt)
                char_names.append(name)

            # --- 3. 提取分镜 ---
            s_list = data.get("storyboard_list") or data.get("storyboard") or []
            for item in s_list:
                if isinstance(item, dict):
                    scene_prompts.append(item.get("prompt", ""))
                    target_obj = item.get("main_character", "")
                    explicit_idx = item.get("ref_image_index", None)
                    
                    found_indices = []

                    # A. 显式索引优先
                    if explicit_idx is not None:
                        if isinstance(explicit_idx, list):
                            found_indices = [int(x) for x in explicit_idx]
                        else:
                            try: found_indices = [int(explicit_idx)]
                            except: pass
                    
                    # B. 名字匹配兜底
                    if not found_indices and target_obj:
                        targets_to_check = []
                        if isinstance(target_obj, list): targets_to_check = target_obj
                        elif isinstance(target_obj, str):
                            if target_obj.lower() != "none": targets_to_check = [target_obj]

                        detected_indices = []
                        for t_str in targets_to_check:
                            t_str = str(t_str)
                            for idx, registered_name in enumerate(char_names):
                                if registered_name in t_str or t_str in registered_name:
                                    if idx not in detected_indices:
                                        detected_indices.append(idx)
                        found_indices = detected_indices

                    # 分配 A/B
                    idx_a = found_indices[0] if len(found_indices) > 0 else -1
                    idx_b = found_indices[1] if len(found_indices) > 1 else -1
                        
                    ref_indices_a.append(idx_a)
                    ref_indices_b.append(idx_b)

            # --- 4. 兜底 ---
            if not char_prompts: char_prompts = [""]
            if not scene_prompts: scene_prompts = [""]
            target_len = len(scene_prompts)
            if len(ref_indices_a) < target_len:
                ref_indices_a.extend([-1] * (target_len - len(ref_indices_a)))
            if len(ref_indices_b) < target_len:
                ref_indices_b.extend([-1] * (target_len - len(ref_indices_b)))

            return (char_prompts, scene_prompts, ref_indices_a, ref_indices_b)

        except Exception as e:
            print(f"Novel Director Error: {e}")
            return (["Error"], ["Error"], [-1], [-1])


class BatchImageSelector:
    """
    节点2：内存级图片提取器
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
    RETURN_NAMES = ("排序后的参考图", )
    INPUT_IS_LIST = True 
    OUTPUT_IS_LIST = (True,) 
    FUNCTION = "select_images"
    CATEGORY = "Novel Director"

    def select_images(self, 所有角色图Batch, 索引列表):
        output_images = []
        if isinstance(所有角色图Batch, list): character_images = 所有角色图Batch[0]
        else: character_images = 所有角色图Batch

        if len(character_images.shape) == 3:
             character_images = character_images.unsqueeze(0)
        
        batch_size = character_images.shape[0]

        for idx in 索引列表:
            try: idx = int(idx)
            except: idx = -1
            
            if idx < 0 or idx >= batch_size:
                # 索引无效给黑图
                blank_img = torch.zeros_like(character_images[0]).unsqueeze(0)
                output_images.append(blank_img)
            else:
                img = character_images[idx]
                output_images.append(img.unsqueeze(0))

        return (output_images, )


class DynamicCharMask:
    """
    节点3：生成角色存在遮罩 (逻辑判断核心)
    根据索引是否为 -1，生成全白或全黑的遮罩Batch。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "参考图Batch": ("IMAGE", ), # 仅用于获取宽高尺寸
                "索引列表": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("存在遮罩(Mask)", )
    INPUT_IS_LIST = True 
    OUTPUT_IS_LIST = (True,) 
    FUNCTION = "generate_mask"
    CATEGORY = "Novel Director"

    def generate_mask(self, 参考图Batch, 索引列表):
        output_masks = []
        
        # 获取图片尺寸
        if isinstance(参考图Batch, list): ref_imgs = 参考图Batch[0]
        else: ref_imgs = 参考图Batch
        
        # 确保 ref_imgs 至少是 (B, H, W, C)
        if len(ref_imgs.shape) == 3: ref_imgs = ref_imgs.unsqueeze(0)
        
        B, H, W, C = ref_imgs.shape
        # MASK 通常是 (H, W) 的 float32，范围 0-1
        
        # 因为我们是 List 输入，索引列表长度可能和 Image Batch 长度不一致
        # 但通常是一致的。这里以索引列表长度为准。
        
        for i, idx in enumerate(索引列表):
            try: idx = int(idx)
            except: idx = -1
            
            # 创建单张遮罩
            if idx == -1:
                # 角色不存在 -> 全黑 (0.0) -> IPAdapter失效
                mask = torch.zeros((H, W), dtype=torch.float32, device=ref_imgs.device)
            else:
                # 角色存在 -> 全白 (1.0) -> IPAdapter生效
                mask = torch.ones((H, W), dtype=torch.float32, device=ref_imgs.device)
            
            output_masks.append(mask)

        return (output_masks, )

NODE_CLASS_MAPPINGS = { 
    "ScriptJSONParser": ScriptJSONParser, 
    "BatchImageSelector": BatchImageSelector,
    "DynamicCharMask": DynamicCharMask
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "ScriptJSONParser": "剧本JSON解析器 (Novel)", 
    "BatchImageSelector": "按索引提取参考图 (Novel)",
    "DynamicCharMask": "生成角色存在遮罩 (Novel)"
}