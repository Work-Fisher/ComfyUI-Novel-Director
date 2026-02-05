import torch
import json
import re
import os
import numpy as np
import soundfile as sf
import folder_paths
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips, VideoFileClip

# ==========================================
# 1. 演员选角 (Casting) - 6人版
# ==========================================
class DirectorCasting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "prev_cast_dict": ("DICT",),
                "char1_name": ("STRING", {"default": "", "multiline": False, "placeholder": "Role Name 1"}),
                "char1_img": ("IMAGE",),
                "char2_name": ("STRING", {"default": "", "multiline": False, "placeholder": "Role Name 2"}),
                "char2_img": ("IMAGE",),
                "char3_name": ("STRING", {"default": "", "multiline": False, "placeholder": "Role Name 3"}),
                "char3_img": ("IMAGE",),
                "char4_name": ("STRING", {"default": "", "multiline": False, "placeholder": "Role Name 4"}),
                "char4_img": ("IMAGE",),
                "char5_name": ("STRING", {"default": "", "multiline": False, "placeholder": "Role Name 5"}),
                "char5_img": ("IMAGE",),
                "char6_name": ("STRING", {"default": "", "multiline": False, "placeholder": "Role Name 6"}),
                "char6_img": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("角色名册(Dict)",)
    FUNCTION = "register_cast"
    CATEGORY = "Novel Director/1. Pre-Production"

    def register_cast(self, prev_cast_dict=None, **kwargs):
        cast = prev_cast_dict.copy() if prev_cast_dict else {}
        for i in range(1, 7):
            name_key = f"char{i}_name"
            img_key = f"char{i}_img"
            name = kwargs.get(name_key)
            img = kwargs.get(img_key)
            if name and name.strip() and img is not None:
                valid_img = img
                if len(valid_img.shape) == 3: valid_img = valid_img.unsqueeze(0)
                cast[name.strip()] = valid_img
        return (cast,)

# ==========================================
# 2A. 剧本加载 (Audio Script) - 极简格式版
# ==========================================
class DirectorAudioScriptLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_json": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "DICT", "STRING")
    RETURN_NAMES = ("角色列表(Role)", "音色提示词(Instruct)", "纯台词(Text)", "总场数", "索引列表(Loop)", "角色字典(Dict)", "角色设定详情(String)")
    OUTPUT_IS_LIST = (True, True, True, False, True, False, False)
    
    FUNCTION = "parse_audio_script"
    CATEGORY = "Novel Director/1. Pre-Production"

    def parse_audio_script(self, audio_json):
        raw = audio_json[0] if isinstance(audio_json, list) else audio_json
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            clean = match.group(0).replace("```json", "").replace("```", "") if match else raw
            clean = clean.replace("\\\n", "\\n")
            data = json.loads(clean)
        except Exception as e:
            print(f"❌ JSON Parse Error: {e}")
            return (["Err"], ["Err"], ["JSON Error"], 1, [0], {}, "JSON解析失败")

        role_list_data = data.get("role_list", [])
        role_map = {} 
        role_desc_str = "" 
        
        for r in role_list_data:
            r_name = r.get("name", "").strip()
            r_inst = r.get("instruct", "")
            r_text = r.get("text", "") 
            
            if r_name: 
                role_map[r_name] = r_inst
                role_desc_str += f"角色：{r_name} 音色：{r_inst} 样本：{r_text}\n"
        
        juben_text = data.get("juben", "")
        raw_lines = [l.strip() for l in juben_text.split('\n') if l.strip()]

        out_roles, out_instructs, out_texts = [], [], []
        default_instruct = "语速中等，清晰自然的朗读声。"

        for line in raw_lines:
            if "：" in line: line = line.replace("：", ":")
            parts = line.split(":", 1)
            if len(parts) == 2:
                r_name = parts[0].strip()
                text_content = parts[1].strip()
                r_instruct = role_map.get(r_name, default_instruct)
                out_roles.append(r_name)
                out_texts.append(text_content)
                out_instructs.append(r_instruct)
            else:
                out_roles.append("旁白")
                out_texts.append(line)
                out_instructs.append(role_map.get("旁白", default_instruct))

        count = len(out_texts)
        if count == 0:
            return (["Empty"], ["Empty"], ["No Content"], 1, [0], {}, "无内容")
            
        print(f"🎙️ [Script] 解析成功: {count} 行 | 角色库: {len(role_map)} 人")
        return (out_roles, out_instructs, out_texts, count, list(range(count)), role_map, role_desc_str)

# ==========================================
# 2B. 分镜加载 (Visual Storyboard)
# ==========================================
class DirectorVisualStoryboardLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "visual_json": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
                "cast_dict": ("DICT",), 
            }
        }
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("画面提示词列表", "角色1图列表", "角色2图列表")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "parse_visual_script"
    CATEGORY = "Novel Director/1. Pre-Production"

    def parse_visual_script(self, visual_json, cast_dict):
        raw = visual_json[0] if isinstance(visual_json, list) else visual_json
        empty_img = torch.zeros((1, 512, 512, 3))
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            clean = match.group(0).replace("```json", "").replace("```", "") if match else raw
            data = json.loads(clean)
        except:
            return (["Error"], [empty_img], [empty_img])

        story_list = data.get("storyboard_list", [])
        prompt_list, img1_list, img2_list = [], [], []

        for item in story_list:
            prompt_list.append(item.get("prompt", "Scene"))
            chars = item.get("main_character", [])
            if isinstance(chars, str): chars = [c.strip() for c in chars.split(",") if c.strip()]
            i1, i2 = empty_img, empty_img
            if len(chars) > 0:
                t = str(chars[0]).strip()
                for n, img in cast_dict.items():
                    if t in n or n in t: i1 = img; break
            if len(chars) > 1:
                t = str(chars[1]).strip()
                for n, img in cast_dict.items():
                    if t in n or n in t: i2 = img; break
            img1_list.append(i1)
            img2_list.append(i2)

        if not prompt_list: return (["Empty"], [empty_img], [empty_img])
        return (prompt_list, img1_list, img2_list)

# ==========================================
# 2C. 视频提示词加载 (Video Prompt Loader)
# ==========================================
class DirectorVideoPromptLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_json": ("STRING", {"multiline": True, "forceInput": True, "default": ""}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频运镜列表(Video Prompts)",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "parse_video_prompt"
    CATEGORY = "Novel Director/1. Pre-Production"

    def parse_video_prompt(self, video_json):
        raw = video_json[0] if isinstance(video_json, list) else video_json
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            clean = match.group(0).replace("```json", "").replace("```", "") if match else raw
            clean = clean.replace("\\\n", "\\n")
            data = json.loads(clean)
        except Exception as e:
            print(f"❌ Video JSON Error: {e}")
            return (["Slow motion"],)

        prompts = data.get("video_prompts", data.get("prompts", []))
        if not isinstance(prompts, list): prompts = ["High quality motion"]
        final_list = [str(p).strip() for p in prompts if p]
        if not final_list: final_list = ["Static camera"]
        
        print(f"🎥 [VideoLoader] Loaded {len(final_list)} motion prompts")
        return (final_list,)

# ==========================================
# 3. 场景处理器 (Iterator)
# ==========================================
class DirectorSceneIterator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scene_index": ("INT", {"forceInput": True}), 
                "role_list": ("STRING", {"forceInput": True}),
                "instruct_list": ("STRING", {"forceInput": True}),
                "text_list": ("STRING", {"forceInput": True}),
                "visual_prompts": ("STRING", {"forceInput": True}),
                "video_prompts": ("STRING", {"forceInput": True}), # 新增
                "char_img1_list": ("IMAGE", ),
                "char_img2_list": ("IMAGE", ),
            }
        }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("当前Role", "Instruct(含Role)", "Text(含Role)", "当前画面Prompt", "当前视频运镜", "Img1", "Img2", "Filename", "Idx_Current", "Idx_Total")
    INPUT_IS_LIST = (True, True, True, True, True, True, True, True)
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, True, True)
    FUNCTION = "process"
    CATEGORY = "Novel Director/2. Production"

    def process(self, scene_index, role_list, instruct_list, text_list, visual_prompts, video_prompts, char_img1_list, char_img2_list):
        def to_list(x): return x if isinstance(x, list) else [x]
        indices, roles, instructs, texts = to_list(scene_index), to_list(role_list), to_list(instruct_list), to_list(text_list)
        vis_prompts, vid_prompts = to_list(visual_prompts), to_list(video_prompts)
        imgs1 = char_img1_list if isinstance(char_img1_list, list) else [char_img1_list]
        imgs2 = char_img2_list if isinstance(char_img2_list, list) else [char_img2_list]
        
        o_role, o_inst, o_txt, o_vis, o_vid = [], [], [], [], []
        o_i1, o_i2, o_fn, o_idx, o_tot = [], [], [], [], []
        
        num_texts = len(texts)
        num_total = len(indices)

        for i in indices:
            idx = int(i)
            t_idx = idx if idx < num_texts else -1
            vis_idx = idx % len(vis_prompts) if len(vis_prompts) > 0 else 0
            vid_idx = idx % len(vid_prompts) if len(vid_prompts) > 0 else 0
            im1_idx = idx % len(imgs1) if len(imgs1) > 0 else 0
            im2_idx = idx % len(imgs2) if len(imgs2) > 0 else 0
            
            raw_role = roles[t_idx]
            final_text = f"{raw_role}: {texts[t_idx]}"
            final_inst = f"{raw_role}: {instructs[t_idx]}"

            o_role.append(raw_role)
            o_inst.append(final_inst)
            o_txt.append(final_text)
            o_vis.append(vis_prompts[vis_idx])
            o_vid.append(vid_prompts[vid_idx])
            o_i1.append(imgs1[im1_idx])
            o_i2.append(imgs2[im2_idx])
            o_fn.append(f"Scene_{idx:03d}")
            o_idx.append(idx)
            o_tot.append(num_total)

        print(f"🔄 [Processor] Batch {num_total} Ready.")
        return (o_role, o_inst, o_txt, o_vis, o_vid, o_i1, o_i2, o_fn, o_idx, o_tot)

# ==========================================
# 3B. 时长计算器 (含缓冲帧)
# ==========================================
class DirectorAudioFrameCalc:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("*", ), 
                "fps": ("INT", {"default": 24}),
                "min_frames": ("INT", {"default": 16}),
                "buffer_frames": ("INT", {"default": 16, "min": 0, "max": 120}),
            }
        }
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("Frames", "Debug_Info")
    FUNCTION = "calc"
    CATEGORY = "Novel Director/2. Production"

    def calc(self, audio, fps, min_frames, buffer_frames):
        aud = audio[0] if isinstance(audio, list) else audio
        dur = 5.0
        if isinstance(aud, str) and os.path.exists(aud):
            try: dur = sf.info(aud).duration
            except: pass
        elif isinstance(aud, dict) and 'waveform' in aud:
            w = aud['waveform']
            if w.dim()==3: w=w.squeeze(0)
            dur = w.shape[-1] / aud['sample_rate']
        
        base_frames = int(dur * fps)
        final_frames = max(base_frames, min_frames) + buffer_frames
        return (final_frames, f"🎞️ {final_frames} frames ({dur:.2f}s + buf) | FPS:{fps}")

# ==========================================
# 4A. 红绿灯
# ==========================================
class DirectorOrderGate:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "wait_for": ("*",), "pass_data": ("*",), } }
    RETURN_TYPES = ("*",) 
    RETURN_NAMES = ("data_output",)
    FUNCTION = "gate"
    CATEGORY = "Novel Director/Logic"
    def gate(self, wait_for, pass_data): return (pass_data,)

# ==========================================
# 4B. 实时存档 (Saver)
# ==========================================
class DirectorStreamSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("IMAGE", ),
                "audio": ("*", ), 
                "filename_prefix": ("STRING", {"default": "scene"}),
                "idx_current": ("INT", {"forceInput": True}),
                "idx_total": ("INT", {"forceInput": True}),
                "fps": ("INT", {"default": 24}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Manifest_Path",)
    OUTPUT_NODE = True
    FUNCTION = "save_scene"
    CATEGORY = "Novel Director/3. Post-Production"

    def save_scene(self, video, audio, filename_prefix, idx_current, idx_total, fps):
        out_dir = os.path.join(folder_paths.get_output_directory(), "Novel_Project")
        os.makedirs(out_dir, exist_ok=True)
        manifest_path = os.path.join(out_dir, "manifest.txt")
        
        curr_idx = idx_current[0] if isinstance(idx_current, list) else idx_current
        if curr_idx == 0 and os.path.exists(manifest_path):
            try: os.remove(manifest_path)
            except: pass

        curr_video = video[0] if isinstance(video, list) else video
        curr_audio = audio[0] if isinstance(audio, list) else audio
        prefix = filename_prefix[0] if isinstance(filename_prefix, list) else filename_prefix
        
        file_name = f"{prefix}.mp4"
        full_path = os.path.join(out_dir, file_name)

        import time
        ts = int(time.time() * 1000)
        temp_audio = os.path.join(out_dir, f"temp_{ts}.wav")
        audio_ready = False
        try:
            if isinstance(curr_audio, str) and os.path.exists(curr_audio):
                data, sr = sf.read(curr_audio)
                sf.write(temp_audio, data, sr)
                audio_ready = True
            elif isinstance(curr_audio, dict) and 'waveform' in curr_audio:
                w = curr_audio['waveform'].squeeze(0)
                if w.shape[0] > w.shape[1]: w = w.t()
                sf.write(temp_audio, w.cpu().numpy().T, curr_audio['sample_rate'])
                audio_ready = True
        except: pass

        v_np = (curr_video.cpu().numpy() * 255).astype(np.uint8)
        frames = [v_np[i] for i in range(v_np.shape[0])] if len(v_np.shape)==4 else [v_np]
        
        try:
            clip = ImageSequenceClip(frames, fps=fps)
            if audio_ready:
                au_clip = AudioFileClip(temp_audio)
                clip = clip.set_audio(au_clip)
            clip.write_videofile(full_path, fps=fps, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
            print(f"💾 Scene {curr_idx} Saved")
        except Exception as e:
            print(f"❌ Save Error: {e}")

        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(full_path + "\n")
        if os.path.exists(temp_audio): os.remove(temp_audio)
        return (manifest_path,)

# ==========================================
# 5. 最终合并 (Final Render)
# ==========================================
class DirectorFinalRender:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "manifest_paths_list": ("STRING", {"forceInput": True}), } }
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("Final_Image", "Final_Audio")
    OUTPUT_NODE = True
    INPUT_IS_LIST = (True,)
    FUNCTION = "render_all"
    CATEGORY = "Novel Director/3. Post-Production"

    def render_all(self, manifest_paths_list):
        print("🎬 [FinalRender] Processing...")
        if not manifest_paths_list: return (torch.zeros(1, 64, 64, 3), None)
        manifest_path = manifest_paths_list[0]
        if not os.path.exists(manifest_path): return (torch.zeros(1, 64, 64, 3), None)
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        video_files = list(dict.fromkeys(lines))
        if not video_files: return (torch.zeros(1, 64, 64, 3), None)

        out_dir = os.path.dirname(manifest_path)
        final_path = os.path.join(out_dir, "Final_Movie_Full.mp4")

        try:
            clips = []
            for vf in video_files:
                if os.path.exists(vf): clips.append(VideoFileClip(vf))
            if clips:
                final = concatenate_videoclips(clips, method="compose")
                final.write_videofile(final_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
                
                # 内存保护：只加载 60 秒以内的视频预览
                if final.duration > 60:
                    frame = final.get_frame(0)
                    frames = [frame]
                else:
                    frames = []
                    for frame in final.iter_frames(): frames.append(frame)
                video_tensor = torch.from_numpy(np.array(frames)).float() / 255.0 if frames else torch.zeros(1, 512, 512, 3)

                audio_out = None
                if final.audio:
                    sr = 44100
                    audio_arr = final.audio.to_soundarray(fps=sr)
                    if audio_arr is not None:
                        waveform = torch.from_numpy(audio_arr.T).float().unsqueeze(0)
                        audio_out = {"waveform": waveform, "sample_rate": sr}
                for c in clips: c.close()
                final.close()
                return (video_tensor, audio_out)
        except Exception: pass
        return (torch.zeros(1, 64, 64, 3), None)

# ==========================================
# 🆕 字典转文本
# ==========================================
class DirectorDictToString:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "data_dict": ("DICT",), } }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Formatted_Text",)
    FUNCTION = "convert"
    CATEGORY = "Novel Director/Logic"
    def convert(self, data_dict):
        try:
            return (json.dumps(data_dict, indent=4, ensure_ascii=False),)
        except: return ("{}",)

NODE_CLASS_MAPPINGS = {
    "DirectorCasting": DirectorCasting,
    "DirectorAudioScriptLoader": DirectorAudioScriptLoader,
    "DirectorVisualStoryboardLoader": DirectorVisualStoryboardLoader,
    "DirectorVideoPromptLoader": DirectorVideoPromptLoader,
    "DirectorSceneIterator": DirectorSceneIterator,
    "DirectorAudioFrameCalc": DirectorAudioFrameCalc,
    "DirectorOrderGate": DirectorOrderGate,
    "DirectorStreamSaver": DirectorStreamSaver,
    "DirectorFinalRender": DirectorFinalRender,
    "DirectorDictToString": DirectorDictToString
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectorCasting": "🎬 1. 演员选角 (6人版)",
    "DirectorAudioScriptLoader": "🎙️ 2A. 有声剧脚本加载 (Audio)",
    "DirectorVisualStoryboardLoader": "🎨 2B. 分镜脚本加载 (Visual)",
    "DirectorVideoPromptLoader": "📹 2C. 视频提示词加载 (Video)",
    "DirectorSceneIterator": "🔄 3. 场景处理器 (Batch Processor)",
    "DirectorAudioFrameCalc": "⏱️ 3B. 时长计算器 (Calc)",
    "DirectorOrderGate": "🚦 强行流控门 (Order Gate)",
    "DirectorStreamSaver": "💾 4. 实时存档 (Saver)",
    "DirectorFinalRender": "🎞️ 5. 最终合并 (Render)",
    "DirectorDictToString": "🔧 字典转文本 (Dict To String)"
}
