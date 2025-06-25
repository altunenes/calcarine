# Calcarine

Scene analysis from PHI-3.5 Vision model.


<div align="center">
  <img width="600" alt="calcarine" src="https://github.com/user-attachments/assets/b0596266-882c-4231-97bd-5deb59e5f79e" />
</div>



## Features

- **Multi-source input**: Images, videos, webcam feeds - We pass textures directly to the GPU
- **Real-time GPU processing**: Compute shaders with hot reload via Cuneus
- **VLM analysis**: Real-time predictions using Microsoft PHI-3.5 Vision model (CPU) via ort
- **Interactive controls**: Real-time parameter adjustment via egui

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## What You're Actually Using

Under the hood, Calcarine is powered by [**Cuneus**](https://github.com/altunenes/cuneus). The real focus is passing textures to GPU and having the VLM make real-time predictions and analysis. The simple image effects (brightness, contrast, etc.) are just examples of what compute shaders can do.

## AI Model Details

**Model**: [Microsoft PHI-3.5 Vision Instruct ONNX](https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4)
- **Optimization**: CPU INT4 quantized for mobile/desktop efficiency
- **Size**: ~3.2 GB (one-time download)
- **Storage**: Models are downloaded to `~/data/3.5_v/` directory
- **Performance**: Tested on MacBook Air M3 16GB - runs smoothly with no issues

*Note: While GPU-optimized VLM models would be ideal, the current ONNX ecosystem has limited options. CPU execution was chosen to avoid the maintenance burden of supporting different execution providers (CUDA, DirectML, CoreML, etc.) across platforms. Even with CPU inference, the model delivers solid results for real-time analysis.*

