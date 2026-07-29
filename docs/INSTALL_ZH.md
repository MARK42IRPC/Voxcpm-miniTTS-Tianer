# Windows 一键安装

## 首次安装

双击 `install_and_start.bat`，选择模型档位。脚本会依次：

1. 下载或复用 `uv`。
2. 安装 Python 3.12 和 `uv.lock` 中锁定的依赖。
3. 下载所选 VoxCPM、ZipEnhancer 和 Piper 学生模型。
4. 启动 `http://127.0.0.1:8810`。

安装器可以重复运行。完整模型会跳过，未完成的 Hugging Face/ModelScope 下载会继续。

## 模型档位

| 档位 | 内容 | 模型下载量（约） |
| --- | --- | ---: |
| 轻量 | VoxCPM 0.5B、ZipEnhancer、Piper 华妍 x_low | 2 GB |
| 推荐 | VoxCPM2、0.5B、ZipEnhancer、两个 Piper 音色 | 8 GB |
| 完整 | 三个 VoxCPM、ZipEnhancer、四个 Piper 音色、MeloTTS 训练基座 | 10 GB |
| 仅依赖 | 只创建 Python 环境 | 0 GB |

依赖与下载缓存还会占用额外空间。默认缓存目录为 `C:\tmp\voxcpm`，可提前设置环境变量 `VOXCPM_CACHE_DIR` 修改。

## 网络选项

使用 Hugging Face 国内镜像：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Profile recommended -UseChinaMirror
```

指定其他 Hugging Face 端点：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Profile recommended -HfEndpoint https://example.com
```

依赖或模型下载失败时，保留已下载文件，修复网络后重新执行相同命令即可。

## 日常启动

安装完成后直接双击 `start_webui.bat`，不会再次执行依赖安装。若 8810 端口已有 WebUI，脚本会直接打开页面。

## 维护命令

仅检查安装器：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Profile recommended -Check
```

强制重新校验模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Profile recommended -ForceModels
```
