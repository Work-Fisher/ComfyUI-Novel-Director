import torch
import json
import re
import os
import time
import numpy as np
import soundfile as sf
import folder_paths
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips, VideoFileClip, CompositeAudioClip, AudioClip, concatenate_audioclips

# ==========================================
# 1. 演员选角 (Casting)
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
# 2A. 剧本加载 (Audio Script)
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
            return (["Err"], ["Err"], ["JSON Error"], 1, [0], {}, "[]")

        role_list_data = data.get("role_list", [])
        role_desc_str = str(role_list_data)
        role_map = {} 
        for r in role_list_data:
            r_name = r.get("name", "").strip()
            r_inst = r.get("instruct", "")
            if r_name: role_map[r_name] = r_inst
        
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
        if count == 0: return (["Empty"], ["Empty"], ["No Content"], 1, [0], {}, "[]")
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
    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("画面提示词列表(Visual)", "角色1图", "角色2图", "角色3图", "角色4图", "角色5图", "角色6图")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True)
    FUNCTION = "parse_visual_script"
    CATEGORY = "Novel Director/1. Pre-Production"

    def parse_visual_script(self, visual_json, cast_dict):
        raw = visual_json[0] if isinstance(visual_json, list) else visual_json
        empty_img = torch.zeros((1, 512, 512, 3))
        
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            clean = match.group(0).replace("```json", "").replace("```", "") if match else raw
            data = json.loads(clean)
        except Exception as e: 
            print(f"❌ Storyboard JSON Error: {e}")
            return (["Error"], [empty_img], [empty_img], [empty_img], [empty_img], [empty_img], [empty_img])

        story_list = data.get("storyboard_list", [])
        prompt_list = []
        
        imgs_cols = [[], [], [], [], [], []] 

        for item in story_list:
            prompt_list.append(item.get("prompt", "Scene"))
            
            chars = item.get("main_character", [])
            if isinstance(chars, str): 
                chars = [c.strip() for c in chars.split(",") if c.strip()]
            
            current_scene_imgs = [empty_img] * 6
            
            for i, char_name in enumerate(chars):
                if i >= 6: break 
                c_name = str(char_name).strip()
                for cast_name, cast_img in cast_dict.items():
                    if c_name in cast_name or cast_name in c_name:
                        current_scene_imgs[i] = cast_img
                        break
            
            for k in range(6):
                imgs_cols[k].append(current_scene_imgs[k])

        if not prompt_list: 
            return (["Empty"], [empty_img], [empty_img], [empty_img], [empty_img], [empty_img], [empty_img])
            
        print(f"🎨 [VisualLoader] Loaded {len(prompt_list)} scenes.")
        return (
            prompt_list, 
            imgs_cols[0], imgs_cols[1], imgs_cols[2], imgs_cols[3], imgs_cols[4], imgs_cols[5]
        )

# ==========================================
# 2C. 视频提示词加载 (Video Prompts)
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
    RETURN_NAMES = ("视频运镜列表(List)",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "parse_video_prompt"
    CATEGORY = "Novel Director/1. Pre-Production"

    def parse_video_prompt(self, video_json):
        raw = video_json[0] if isinstance(video_json, list) else video_json
        default_prompts = ["Static camera, high quality"]

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            clean = match.group(0).replace("```json", "").replace("```", "") if match else raw
            clean = clean.replace("\\\n", "\\n")
            data = json.loads(clean)
            
            prompts = data.get("video_prompts", data.get("prompts", []))
            
            if not isinstance(prompts, list):
                print("⚠️ Video JSON warning: 'video_prompts' is not a list. Using default.")
                return (default_prompts,)
            
            final_list = [str(p).strip() for p in prompts if p]
            if not final_list: return (default_prompts,)
                
            print(f"📹 [VideoLoader] 成功加载 {len(final_list)} 条独立视频提示词。")
            return (final_list,)

        except Exception as e:
            print(f"❌ Video JSON Parse Error: {e}")
            return (default_prompts,)

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
                "video_prompts": ("STRING", {"forceInput": True}), 
                "char_img1_list": ("IMAGE", ),
                "char_img2_list": ("IMAGE", ),
            },
            "optional": {
                "char_img3_list": ("IMAGE", ),
                "char_img4_list": ("IMAGE", ),
                "char_img5_list": ("IMAGE", ),
                "char_img6_list": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("当前Role", "Instruct(含Role)", "Text(含Role)", "当前画面Prompt", "当前视频运镜(Single)", "Img1", "Img2", "Img3", "Img4", "Img5", "Img6", "Filename", "Idx_Current", "Idx_Total")
    
    INPUT_IS_LIST = (True, True, True, True, True, True, True, True, True, True, True, True)
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    
    FUNCTION = "process"
    CATEGORY = "Novel Director/2. Production"

    def process(self, scene_index, role_list, instruct_list, text_list, visual_prompts, video_prompts, 
                char_img1_list, char_img2_list, 
                char_img3_list=None, char_img4_list=None, char_img5_list=None, char_img6_list=None):
        
        def to_list(x): return x if isinstance(x, list) else [x]
        
        indices = to_list(scene_index)
        roles = to_list(role_list)
        instructs = to_list(instruct_list)
        texts = to_list(text_list)
        vis_prompts = to_list(visual_prompts)
        vid_prompts_source = to_list(video_prompts)

        empty_tensor = torch.zeros((1, 512, 512, 3))
        def get_img_list(l): return l if (l is not None and isinstance(l, list)) else [empty_tensor]
        
        imgs_matrix = [
            get_img_list(char_img1_list), get_img_list(char_img2_list),
            get_img_list(char_img3_list), get_img_list(char_img4_list),
            get_img_list(char_img5_list), get_img_list(char_img6_list)
        ]
        
        o = {
            "role": [], "inst": [], "txt": [], "vis": [], "vid": [],
            "i1": [], "i2": [], "i3": [], "i4": [], "i5": [], "i6": [],
            "fn": [], "idx": [], "tot": []
        }
        
        num_texts = len(texts)
        num_total = len(indices)
        num_vis = len(vis_prompts)
        num_vid = len(vid_prompts_source)
        
        for i in indices:
            idx = int(i)
            t_idx = idx if idx < num_texts else -1
            vis_idx = idx % num_vis if num_vis > 0 else 0
            vid_idx = idx % num_vid if num_vid > 0 else 0
            current_single_vid_prompt = vid_prompts_source[vid_idx]
            
            img_indices = [idx % len(lst) if len(lst) > 0 else 0 for lst in imgs_matrix]
            
            o["role"].append(roles[t_idx])
            o["inst"].append(f"{roles[t_idx]}: {instructs[t_idx]}")
            o["txt"].append(f"{roles[t_idx]}: {texts[t_idx]}")
            
            o["vis"].append(vis_prompts[vis_idx])
            o["vid"].append(current_single_vid_prompt)
            
            o["i1"].append(imgs_matrix[0][img_indices[0]])
            o["i2"].append(imgs_matrix[1][img_indices[1]])
            o["i3"].append(imgs_matrix[2][img_indices[2]])
            o["i4"].append(imgs_matrix[3][img_indices[3]])
            o["i5"].append(imgs_matrix[4][img_indices[4]])
            o["i6"].append(imgs_matrix[5][img_indices[5]])
            
            o["fn"].append(f"Scene_{idx:03d}")
            o["idx"].append(idx)
            o["tot"].append(num_total)

        print(f"🔄 [Processor] Processed {len(indices)} frames.")

        return (
            o["role"], o["inst"], o["txt"], o["vis"], o["vid"],
            o["i1"], o["i2"], o["i3"], o["i4"], o["i5"], o["i6"],
            o["fn"], o["idx"], o["tot"]
        )

# ==========================================
# 3B. 时长计算器 (Calc)
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
        # 这里计算出的总帧数包含了Buffer，所以视频会比原始音频长
        final_frames = max(base_frames, min_frames) + buffer_frames
        return (final_frames, f"🎞️ {final_frames} frames ({dur:.2f}s + buf) | FPS:{fps}")

# ==========================================
# 4A. 逻辑控制门 (Gate)
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
# 4B. 实时存档 (Saver) - [★核心优化：精确对齐音频]
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
        # 如果是第一帧，清理旧清单
        if curr_idx == 0 and os.path.exists(manifest_path):
            try: os.remove(manifest_path)
            except: pass

        curr_video = video[0] if isinstance(video, list) else video
        curr_audio = audio[0] if isinstance(audio, list) else audio
        prefix = filename_prefix[0] if isinstance(filename_prefix, list) else filename_prefix
        
        file_name = f"{prefix}.mp4"
        full_path = os.path.join(out_dir, file_name)
        
        # 1. 准备视频帧
        v_np = (curr_video.cpu().numpy() * 255).astype(np.uint8)
        frames = [v_np[i] for i in range(v_np.shape[0])] if len(v_np.shape)==4 else [v_np]
        
        # ★ 计算视频理论精确时长
        video_duration_sec = len(frames) / float(fps)
        
        # 2. 处理音频
        ts = int(time.time() * 1000)
        temp_audio = os.path.join(out_dir, f"temp_audio_{curr_idx}_{ts}.wav")
        audio_ready = False
        
        try:
            # 统一转为 numpy
            audio_data = None
            sample_rate = 44100
            
            if isinstance(curr_audio, str) and os.path.exists(curr_audio):
                audio_data, sample_rate = sf.read(curr_audio)
            elif isinstance(curr_audio, dict) and 'waveform' in curr_audio:
                w = curr_audio['waveform']
                if w.dim() == 3: w = w.squeeze(0) 
                sample_rate = curr_audio['sample_rate']
                if w.shape[0] < w.shape[1]: w = w.t()
                audio_data = w.cpu().numpy()
            
            if audio_data is not None:
                # ★ 强制音频长度对齐视频长度 (补静音或截断)
                target_samples = int(video_duration_sec * sample_rate)
                current_samples = len(audio_data)
                
                if current_samples < target_samples:
                    # 需要补静音 (这就是你想要的效果：延长画面对应静音)
                    pad_len = target_samples - current_samples
                    if audio_data.ndim == 2:
                        padding = np.zeros((pad_len, audio_data.shape[1]), dtype=audio_data.dtype)
                        final_audio_np = np.vstack((audio_data, padding))
                    else:
                        padding = np.zeros(pad_len, dtype=audio_data.dtype)
                        final_audio_np = np.concatenate((audio_data, padding))
                else:
                    # 如果音频比视频长(罕见)，截断
                    final_audio_np = audio_data[:target_samples]
                
                sf.write(temp_audio, final_audio_np, sample_rate)
                audio_ready = True
                
        except Exception as e:
            print(f"⚠️ [Saver] Audio Align Error: {e}")

        clip = None
        au_clip = None
        
        try:
            clip = ImageSequenceClip(frames, fps=fps)
            
            if audio_ready and os.path.exists(temp_audio):
                au_clip = AudioFileClip(temp_audio)
                # 再次强制设置时长，确保 MoviePy 不会混乱
                au_clip = au_clip.set_duration(video_duration_sec)
                clip = clip.set_audio(au_clip)

            clip.write_videofile(
                full_path, 
                fps=fps, 
                codec="libx264", 
                audio_codec="aac", 
                audio_fps=44100, 
                preset="ultrafast", 
                logger=None, 
                ffmpeg_params=["-pix_fmt", "yuv420p"]
            )
            print(f"💾 Scene {curr_idx} Saved: {file_name} (Dur: {video_duration_sec:.2f}s | FPS: {fps})")

        except Exception as e:
            print(f"❌ [Saver] Write Error: {e}")
        finally:
            try:
                if clip: clip.close()
                if au_clip: au_clip.close()
                if os.path.exists(temp_audio): 
                    time.sleep(0.1)
                    os.remove(temp_audio)
            except: pass

        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(full_path + "\n")
        return (manifest_path,)

# ==========================================
# 5. 最终合并 (Render) - [★核心优化：统一FPS与预览对齐]
# ==========================================
class DirectorFinalRender:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "manifest_paths_list": ("STRING", {"forceInput": True}),
                # ★ 新增：必须传入 FPS，使其与 Calc 和 Saver 保持一致
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}), 
            } 
        }
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("Final_Image", "Final_Audio")
    OUTPUT_NODE = True
    INPUT_IS_LIST = (True, False) # manifest是List，fps是单个值
    FUNCTION = "render_all"
    CATEGORY = "Novel Director/3. Post-Production"

    def render_all(self, manifest_paths_list, fps):
        print(f"🎬 [FinalRender] Processing with target FPS: {fps}...")
        
        # 处理输入是列表的情况
        target_fps = fps[0] if isinstance(fps, list) else fps
        
        if not manifest_paths_list: return (torch.zeros(1, 64, 64, 3), None)
        manifest_path = manifest_paths_list[0]
        if not os.path.exists(manifest_path): return (torch.zeros(1, 64, 64, 3), None)
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        video_files = list(dict.fromkeys(lines))
        if not video_files: return (torch.zeros(1, 64, 64, 3), None)

        out_dir = os.path.dirname(manifest_path)
        final_path = os.path.join(out_dir, "Final_Movie_Full.mp4")
        debug_audio_path = os.path.join(out_dir, "Final_Audio_Debug.wav")
        TARGET_SR = 44100

        try:
            clips = []
            for vf in video_files:
                if os.path.exists(vf): 
                    # 使用 target_fps 加载，确保对其
                    clips.append(VideoFileClip(vf, target_resolution=None))
            
            if clips:
                # 1. 物理合并
                final = concatenate_videoclips(clips, method="compose")
                
                # 写入文件时严格使用传入的 FPS
                final.write_videofile(
                    final_path, 
                    fps=target_fps, 
                    codec="libx264", 
                    audio_codec="aac", 
                    audio_fps=TARGET_SR,
                    preset="ultrafast",
                    logger=None,
                    ffmpeg_params=["-pix_fmt", "yuv420p"]
                )
                
                if final.audio:
                    final.audio.write_audiofile(debug_audio_path, fps=TARGET_SR, logger=None)
                
                final.close() 
                for c in clips: c.close()
                time.sleep(0.5) 

                # 2. 重载视频用于 ComfyUI 预览
                print(f"🔄 Reloading from {final_path} for preview...")
                video_tensor = torch.zeros(1, 512, 512, 3)
                num_frames = 0
                
                if os.path.exists(final_path):
                    import imageio
                    reader = imageio.get_reader(final_path)
                    frames = []
                    for frame in reader:
                        frames.append(frame)
                    reader.close()
                    
                    if frames:
                        num_frames = len(frames)
                        video_tensor = torch.from_numpy(np.array(frames)).float() / 255.0
                
                # 3. 重载音频并对齐
                audio_out = None
                target_audio_file = debug_audio_path if os.path.exists(debug_audio_path) else final_path
                
                try:
                    if os.path.exists(target_audio_file) and num_frames > 0:
                        data, sr = sf.read(target_audio_file)
                        
                        # ★ 核心修正：基于传入的 FPS 计算视频时长，而不是默认值
                        vid_duration = num_frames / float(target_fps)
                        curr_samples = len(data) if data.ndim == 1 else data.shape[0]
                        aud_duration = curr_samples / sr
                        
                        print(f"📊 Sync Check: Video({vid_duration:.2f}s) vs Audio({aud_duration:.2f}s) at {target_fps} FPS")

                        # 只有当误差真正很大时才修剪，避免因帧率计算导致的微小误差被错误裁剪
                        expected_samples = int(vid_duration * sr)
                        
                        # 如果差异超过 2 帧的时长，则进行强制对齐
                        frame_dur = 1.0 / target_fps
                        if abs(vid_duration - aud_duration) > (frame_dur * 2):
                            print(f"⚠️ Detected Desync, forcing alignment to video length.")
                            if curr_samples > expected_samples:
                                data = data[:expected_samples]
                            else:
                                pad_len = expected_samples - curr_samples
                                if data.ndim == 1: padding = np.zeros(pad_len)
                                else: padding = np.zeros((pad_len, data.shape[1]))
                                data = np.concatenate((data, padding))
                        
                        if len(data) > 0:
                            if data.ndim == 1: data = data[:, np.newaxis]
                            data = data.T 
                            waveform = torch.from_numpy(data).float().unsqueeze(0)
                            audio_out = {"waveform": waveform, "sample_rate": sr}
                            
                except Exception as e:
                    print(f"❌ Audio Load Error: {e}")
                    
                return (video_tensor, audio_out)
        except Exception as e:
            print(f"❌ Render Error: {e}")
            import traceback
            traceback.print_exc()
        
        return (torch.zeros(1, 64, 64, 3), None)

# ==========================================
# 6. 工具节点 (DictToString)
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
        try: return (json.dumps(data_dict, indent=4, ensure_ascii=False),)
        except: return ("{}",)

# ==========================================
# 节点映射
# ==========================================
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
    "DirectorVideoPromptLoader": "📹 2C. 视频提示词加载 (独立JSON)",
    "DirectorSceneIterator": "🔄 3. 场景处理器 (6人+视频拆分)",
    "DirectorAudioFrameCalc": "⏱️ 3B. 时长计算器 (Calc)",
    "DirectorOrderGate": "🚦 强行流控门 (Order Gate)",
    "DirectorStreamSaver": "💾 4. 实时存档 (Saver)",
    "DirectorFinalRender": "🎞️ 5. 最终合并 (Render)",
    "DirectorDictToString": "🔧 字典转文本 (Dict To String)"
}
