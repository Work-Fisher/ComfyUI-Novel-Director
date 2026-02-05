🎬 ComfyUI-Novel-Director (AI 有声小说/短剧导演)

ComfyUI-Novel-Director 是一套专为 AI 短剧、有声小说和动态漫制作设计的 ComfyUI 自定义节点包。

它解决了传统工作流中“音画对齐难”、“角色一致性差”、“批量生成繁琐”以及“后期合并累”的痛点。只需喂给它 LLM (如 ChatGPT/DeepSeek) 生成的 JSON 脚本，它就能自动完成从分镜解析、角色分配、语音合成驱动到最终视频合并的全流程。

✨ 核心功能

👥 多角色统筹 (Casting System)：支持单节点 6 人选角，并可通过串联无限扩展。支持图像与角色名绑定。

📜 智能脚本解析:

Audio Script: 自动拆分剧本，支持提取角色音色（Instruct）、台词（Text）和角色名。

Visual Storyboard: 解析分镜画面提示词。

Video Prompts: 解析运镜/动态提示词。

🔄 自动流式处理 (Auto Batch)：独创的 Scene Iterator 机制，自动对齐文本、语音、画面和运镜，实现“一键生成 100 集”。

🔌 高级 TTS 适配:

自动为台词添加 角色名: 前缀（适配 Qwen-TTS 等大模型）。

输出角色字典（Dict）供 GPT-SoVITS 等模型建立音色库。

生成极简文本格式的角色设定表，方便预览。

⏱️ 精准时序控制: 根据音频时长自动计算视频帧数，支持设置 Buffer Frames (缓冲帧)，让画面切换具有呼吸感。

🎞️ 自动合并与回流: 所有片段生成完毕后，自动合并为 Final_Movie.mp4，并支持将长视频回传给 ComfyUI (Video Combine) 进行二次处理。

📦 安装说明

进入你的 ComfyUI 插件目录：

code
Bash
download
content_copy
expand_less
cd ComfyUI/custom_nodes/

克隆本仓库：

code
Bash
download
content_copy
expand_less
git clone https://github.com/Work-Fisher/ComfyUI-Novel-Director.git

安装依赖：

code
Bash
download
content_copy
expand_less
cd ComfyUI-Novel-Director
pip install -r requirements.txt

重启 ComfyUI。

🧩 节点详解
1. Pre-Production (前期筹备)

🎬 1. 演员选角 (6人版)

输入角色名字和对应的参考图（用于 IP-Adapter 固定人脸）。

支持 prev_cast_dict 输入，可串联多个节点实现无限角色管理。

🎙️ 2A. 有声剧脚本加载 (Audio)

输入：LLM 生成的剧本 JSON。

输出：

角色列表、提示词、台词：用于驱动 TTS。

角色字典(Dict)：用于高级 TTS 建立音色库。

角色设定详情(String)：可视化文本，方便检查。

🎨 2B. 分镜脚本加载 (Visual)

输入：LLM 生成的分镜 JSON 和 角色名册。

输出：按顺序排列的画面提示词、当前场景对应的角色图片（自动匹配）。

📹 2C. 视频提示词加载 (Video)

输入：LLM 生成的运镜 JSON。

输出：视频生成模型的 Motion Prompts。

2. Production (生产制作)

🔄 3. 场景处理器 (Batch Processor)

核心节点。接收所有脚本列表，将其“压扁”为逐个场景的数据流。

自动为 Text 和 Instruct 拼接 Role: 前缀。

向下游分发：当前台词、当前提示词、当前运镜、当前角色图。

⏱️ 3B. 时长计算器 (Calc)

输入：生成的音频。

设置：FPS 和 Buffer Frames (缓冲帧)。

功能：计算视频需要生成多少帧。缓冲帧可让画面在语音结束后多停留一会（静音），优化观感。

3. Post-Production (后期合成)

💾 4. 实时存档 (Saver)

保存每一个单独的场景片段 (Scene_000.mp4 ...)。

处理音画同步，若视频长于音频（因缓冲帧），自动填充静音。

生成合并清单 (manifest.txt)。

🎞️ 5. 最终合并 (Render)

等待所有片段保存完毕后触发。

利用 ffmpeg/moviepy 无损合并所有片段。

输出最终长视频，并支持回传给 ComfyUI 预览。

📝 JSON 格式参考

请将以下 Prompt 发送给 ChatGPT 或 DeepSeek 来生成脚本：

1. 剧本 (Audio JSON)
code
JSON
download
content_copy
expand_less
{
  "role_list": [
    {"name": "旁白", "instruct": "深沉男声，语速适中", "text": "故事开始了"},
    {"name": "李明", "instruct": "年轻男声，语气激动", "text": "快看那边！"}
  ],
  "juben": "旁白:天色渐晚。\n李明:我们得赶紧回家。\n旁白:两人加快了脚步。"
}
2. 画面分镜 (Visual JSON)
code
JSON
download
content_copy
expand_less
{
  "storyboard_list": [
    {"prompt": "Dark sky, evening, street light", "main_character": []},
    {"prompt": "Two boys running on the street", "main_character": ["李明"]}
  ]
}
3. 运镜 (Video JSON) - 可选
code
JSON
download
content_copy
expand_less
{
  "video_prompts": [
    "Static camera, cinematic lighting",
    "Camera tracking shot, running motion"
  ]
}
🚀 快速开始

搭建工作流：Audio Loader -> Iterator -> TTS / KSampler / VideoGen -> Saver -> Render。

确保 Iterator 的 scene_index 连接自 Audio Loader 的 索引列表(Loop)。

点击 Queue Prompt，观察控制台，它将自动循环执行直到整部剧集制作完成。

⚠️ 常见问题

生成的视频黑屏？

如果最终合并的视频过长（>1分钟），为了防止显存溢出，Final Render 节点可能只返回第一帧作为预览，但完整视频已成功保存在 output/Novel_Project 文件夹下。

如何增加角色？

串联多个 1. 演员选角 节点即可。

License

MIT License. Feel free to use and modify!
