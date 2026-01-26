# ComfyUI Novel Director 🎬

**Novel Director** 是一个专为 AI 故事/漫画生成设计的 ComfyUI 插件。它通过解析 JSON 剧本，自动调度多角色的参考图（Reference Images）和提示词，完美解决多人物场景下的自动化控制问题。

## ✨ 核心功能

1.  **智能剧本解析**：支持通过 JSON 定义角色库和分镜表，自动匹配角色名称。
2.  **多角色调度**：自动识别单人/双人场景，分别输出主角（A）和配角（B）的参考图索引。
3. **Batch 批量处理**：完美支持 ComfyUI 的 Batch 列表模式，一次生成整本分镜。

## 📥 安装方法

进入你的 ComfyUI 插件目录：
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/你的用户名/ComfyUI-Novel-Director.git

重启 ComfyUI 即可。

📝 剧本 JSON 格式

提供两个剧本（需要接合QWEN-TTS一起使用）

1.小说自动转角色音色、台词模板
2.台词转人设、分镜图模板

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



