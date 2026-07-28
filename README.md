<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM" width="180">
  <h1>VoxCPM miniTTS Tianer</h1>
  <p>面向消费级显卡的角色语音克隆、微调与轻量学生模型蒸馏工作台</p>
</div>

> 本仓库是基于 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 的社区扩展，重点解决 Windows 本地安装、4 GB 显存推理、角色音色微调和轻量 ONNX 模型生产。模型能力、论文和官方基准请以原项目为准。

## 快速开始

支持 Windows 10/11。首次使用：

1. 克隆或下载本仓库。
2. 双击 `install_and_start.bat`。
3. 选择模型档位，等待依赖和模型下载完成。
4. 浏览器打开 `http://127.0.0.1:8810`。

安装完成后，日常使用只需双击 `start_webui.bat`。安装器可重复运行，完整的依赖和模型会自动跳过，未完成的下载可以继续。

详细网络、缓存和维护选项见 [Windows 一键安装说明](INSTALL_ZH.md)。

## 这套工作台能做什么

| 工作区 | 主要能力 |
| --- | --- |
| 推理 | 文本转语音、参考音频克隆、声音控制、普通/批量文本任务、LoRA 加载、实时耗时、会话音频列表 |
| 推理优化 | CPU、CUDA、稳定混合、极限混合、输入缓存、GPU DiT、去噪器和模型常驻 |
| 音频后处理 | 响度、增益、压缩、均衡等独立开关；保留原音频与处理版本进行对比 |
| LoRA 训练 | 数据集检查、模型切换、训练轮数、检查点保存、继续训练、暂停释放显存、检查点试听 |
| 学生模型蒸馏 | Piper 与 MeloTTS 架构切换、精度可选微调、检查点管理、试听、ONNX/INT8 导出 |
| 数据闭环 | 将生成音频移动到训练集，根据音频元数据自动生成对应 LAB 文本并检测重复内容 |

生成的 WAV 会写入可复现参数元数据，并使用时间戳命名。后处理版本额外带有 `-af-xxxxx` 哈希后缀。

## 安装档位

| 档位 | 安装内容 | 模型下载量（约） | 适用场景 |
| --- | --- | ---: | --- |
| 轻量 | VoxCPM 0.5B、ZipEnhancer、Piper 华妍 x_low | 2 GB | 快速体验、低资源推理 |
| 推荐 | VoxCPM2、0.5B、ZipEnhancer、两个 Piper 音色 | 8 GB | 4 GB 显存 + 32 GB 内存工作站 |
| 完整 | 三个 VoxCPM、四个 Piper 音色、MeloTTS 训练基座 | 10 GB | 推理、LoRA、学生模型训练全部使用 |
| 仅依赖 | Python 3.12 和锁定依赖 | 0 GB | 自行管理模型文件 |

Python、CUDA 运行库和下载缓存会占用额外空间。推荐至少预留 25 GB，完整安装建议预留 30 GB。

## 硬件建议

当前主要实测设备：NVIDIA GeForce RTX 3050 Laptop GPU 4 GB，系统内存 32 GB。

| 模型或任务 | 4 GB 显存建议 | 说明 |
| --- | --- | --- |
| VoxCPM 0.5B 推理 | CUDA | 速度和显存压力最低，也支持 CPU 推理 |
| VoxCPM 1.5 推理 | CUDA | 4 GB 实测可运行；首次优化器编译可能需要数分钟 |
| VoxCPM2 推理 | 稳定混合 | 主要计算放在 GPU，大模块在 CPU 保持精度；需要较多系统内存 |
| VoxCPM2 全 GPU | 不建议 4 GB | 建议 8 GB 及以上显存 |
| VoxCPM 0.5B LoRA | CUDA，批大小 1 | 当前设备的主力音色微调方案 |
| VoxCPM 1.5 LoRA | CUDA，极限配置 | 4 GB 可尝试，训练速度和余量取决于样本长度 |
| VoxCPM2 LoRA | 不建议 4 GB | WebUI 在显存不足时会阻止启动 |
| Piper / MeloTTS ONNX | CPU | 面向边缘部署，模型体积和依赖较小 |

不同驱动、音频长度和 LoRA 目标层会改变显存占用。开始长任务前建议先用少量数据保存并试听第一个检查点。

## 三个页面

### 推理

入口：`http://127.0.0.1:8810/`

- 选择 VoxCPM2、VoxCPM1.5 或 VoxCPM 0.5B。
- 使用文本转语音、参考音频克隆或极致克隆。
- 在稳定混合模式下复用参考音频输入缓存。
- 普通模式按句末标点分段推理并合成为一个 WAV；批量模式按非空行输出多个 WAV。
- 为参考音频、原始结果和后处理结果提供页面内试听。
- 将满意结果直接移动到所选训练集。

### LoRA 训练

入口：`http://127.0.0.1:8810/lora`

- 扫描 WAV/LAB 训练集并显示样本数、总时长和音频规格。
- 通过训练轮数表达任务长度，日志每个批次都会输出进度。
- 支持选择已有工程继续训练，以及暂停任务释放显存进行试听。
- 推理页面可以刷新并选择兼容当前 VoxCPM 模型的 LoRA。

### 学生模型蒸馏

入口：`http://127.0.0.1:8810/distill`

- Piper：训练、试听、检查点导出和模型管理。
- MeloTTS：使用官方中文基座进行中文或中英混合微调。
- MeloTTS 默认使用 FP32，也可选择 BF16/FP16；loss 或梯度出现 NaN/Inf 时立即停止且不保存无效检查点。
- MeloTTS 检查点可试听并导出约 50 MB 的 INT8 ONNX 部署包。
- 学生模型按架构分类，便于以后接入更多轻量 TTS 引擎。

## 模型与运行目录

以下目录不会提交到 Git：

```text
pretrained_models/   VoxCPM 与 ZipEnhancer 模型
piper/models/        Piper、MeloTTS ONNX 部署模型
piper/runs/          学生模型训练检查点
piper/melo-bases/    MeloTTS 官方训练基座
lora/                本机 LoRA 工程、日志和检查点
outputs/             生成和后处理音频
```

编译和模型下载缓存默认位于 `C:\tmp\voxcpm`。可以在安装前设置 `VOXCPM_CACHE_DIR` 更改位置。

请勿把训练集、参考音频、LoRA 权重或角色语音生成结果直接提交到公开仓库。

## 常见问题

### 首次推理等待很久

启用模型优化时，首次编译可能需要数分钟。编译缓存完成后，后续请求通常会明显加快。页面计时器会区分浏览器等待和服务端处理时间。

### 2B 模型在 4 GB 显存上报错

选择“稳定混合”而不是全 GPU。关闭其他占用显存的软件，并保留足够系统内存。极限混合会放置更多模块到 GPU，但更容易出现数值漂移。

### 参考音频模式输出了参考文案

确认选择的是正确克隆模式。仅参考音频克隆不要求参考文本；极致克隆才会使用参考文本。不要将声音描述手动拼接到生成文本前。

### LoRA 无法加载

LoRA 必须与训练时使用的 VoxCPM 架构匹配。部分编译优化可能影响动态加载，出现问题时先关闭“模型优化”进行排查。

### 下载失败

安装器保留已完成文件。修复代理后重新双击安装器，或使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Profile recommended -UseChinaMirror
```

## 开发与维护

同步依赖并安装开发工具：

```powershell
uv sync --frozen --python 3.12 --extra dev
```

运行测试：

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -X utf8 -m pytest -q tests
```

本机当前回归结果为 `94 passed`。

仓库远端建议保持：

```text
origin    个人维护仓库
upstream  OpenBMB/VoxCPM 原项目
```

## 上游项目与许可证

| 项目 | 用途 | 许可证/链接 |
| --- | --- | --- |
| OpenBMB VoxCPM | 基础模型、推理与微调代码 | [项目](https://github.com/OpenBMB/VoxCPM) · [模型](https://huggingface.co/openbmb) · Apache-2.0 |
| MeloTTS | 中英学生模型训练与导出 | [项目](https://github.com/myshell-ai/MeloTTS) · MIT |
| Piper | 轻量 VITS 训练和 ONNX 语音 | [项目](https://github.com/OHF-Voice/piper1-gpl) · GPL-3.0 |
| sherpa-onnx | MeloTTS ONNX 运行时 | [项目](https://github.com/k2-fsa/sherpa-onnx) · Apache-2.0 |
| ZipEnhancer | 参考音频降噪 | [模型](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base) · Apache-2.0 |

仓库主体沿用 [Apache-2.0](LICENSE)。vendored MeloTTS 源码保留其独立 [MIT 许可证](third_party/MeloTTS/LICENSE) 和上游提交记录。模型权重由安装器从各自来源下载，其许可证以模型发布页为准。

## 使用边界

声音克隆和角色音色训练可能涉及著作权、表演者权、人格权和平台规则。只处理你有权使用的数据，并在公开发布合成音频时明确标注其为 AI 生成内容。本项目不附带任何角色音色数据或训练权重。
