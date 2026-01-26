# ComfyUI Novel Director 🎬

**Novel Director** 是一个专为 AI 故事/漫画生成设计的 ComfyUI 插件。它通过解析 JSON 剧本，自动调度多角色的参考图（Reference Images）和提示词，完美解决多人物场景下的 IPAdapter 自动化控制问题。

## ✨ 核心功能

1.  **智能剧本解析**：支持通过 JSON 定义角色库和分镜表，自动匹配角色名称。
2.  **多角色调度**：自动识别单人/双人场景，分别输出主角（A）和配角（B）的参考图索引。
3.  **动态遮罩逻辑 (Dynamic Masking)**：
    *   当分镜中只有一个人时，配角通道会自动生成**全黑遮罩**，彻底关闭对应的 IPAdapter/ControlNet 影响，防止画面崩坏。
    *   当分镜中两人都在时，生成全白遮罩，激活 IPAdapter。
4.  **Batch 批量处理**：完美支持 ComfyUI 的 Batch 列表模式，一次生成整本分镜。

## 📥 安装方法

进入你的 ComfyUI 插件目录：
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/你的用户名/ComfyUI-Novel-Director.git

重启 ComfyUI 即可。

📝 剧本 JSON 格式

在“剧本JSON解析器”节点中填入如下格式的数据：

code
JSON
download
content_copy
expand_less
{
  "character_list": [
    {
      "name": "Alice",
      "prompt": "1girl, blonde hair, blue eyes, white dress",
      "ref_image_index": 0 
      // 对应上传的角色图 Batch 中的第 0 张
    },
    {
      "name": "Bob",
      "prompt": "1boy, black hair, hoodie",
      "ref_image_index": 1
      // 对应上传的角色图 Batch 中的第 1 张
    }
  ],
  "storyboard_list": [
    {
      "prompt": "Alice is eating an apple under a tree.",
      "main_character": "Alice" 
      // 只写 Alice，配角通道会自动输出全黑遮罩
    },
    {
      "prompt": "Alice and Bob are talking closely.",
      "main_character": ["Alice", "Bob"]
      // 同时识别两人，提取器A输出Alice图，提取器B输出Bob图
    },
    {
      "prompt": "A landscape of a city, empty street.",
      "main_character": "None"
      // 无人场景，两个通道均失效
    }
  ]
}
🔌 节点连接指南 (Workflow)
1. 基础准备

Load Image Batch: 加载一张包含所有角色参考图的大图（或通过 Batch 节点合并），按顺序排列（0: Alice, 1: Bob...）。

JSON Script: 填入上面的 JSON。

2. 连接逻辑

该插件包含三个核心节点：

剧本JSON解析器 (Novel)

输入：JSON 字符串

输出：提示词、主角索引(A)、配角索引(B)

按索引提取参考图 (Novel) (需复制两份)

提取器 A: 连接 主角索引(A) -> 输出图片给 IPAdapter A (image)。

提取器 B: 连接 配角索引(B) -> 输出图片给 IPAdapter B (image)。

生成角色存在遮罩 (Novel) (关键!)

连接 配角索引(B) 和 提取器B的图片（用于取尺寸）。

输出 Mask: 连接到 IPAdapter B 的 attention_mask (或 mask 输入)。

作用：当只有 Alice 时，Bob 的 IPAdapter 会收到全黑 Mask，从而不产生任何干扰。

🛠️ Credits


