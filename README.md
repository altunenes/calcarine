# 🎨 Calcarine

**Real-time VLM analysis via GPU compute shaders**


<img width="1260" alt="calcarine" src="https://github.com/user-attachments/assets/b0596266-882c-4231-97bd-5deb59e5f79e" />


Calcarine demonstrates real-time VLM analysis by passing visual content through GPU compute shaders to Microsoft's PHI-3.5 Vision model for intelligent scene understanding.

## ✨ Features

- **🖼️ Multi-source input**: Images, videos, webcam feeds -> We pass textures directly to the GPU
- **⚡ Real-time GPU processing**: compute shaders with hot reload: Cuneus
- **🤖 VLM analysis**: Real-time predictions using Microsoft PHI-3.5 Vision model (CPU): ort
- **🎛️ Interactive controls**: Real-time parameter adjustment: egui

## 🚀 Quick Start

1. **Install GStreamer+** from [gstreamer.freedesktop.org](https://gstreamer.freedesktop.org/download/)
2. **Extract and run** - AI models download automatically on first launch (~3.2 GB)
3. **Press 'H'** to toggle the settings UI
4. **Drag & drop** media files or enable webcam

## 🎯 What You're Actually Using

Under the hood, Calcarine is powered by the **Cuneus**. The real focus is passing textures to GPU and having the VLM make real-time predictions and analysis. The simple image effects (brightness, contrast, etc.) are just examples of what compute shaders can do.

## 🧠 AI Model Details

**Model**: [Microsoft PHI-3.5 Vision Instruct ONNX](https://huggingface.co/microsoft/Phi-3.5-vision-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4)
- **Optimization**: CPU INT4 quantized for mobile/desktop efficiency
- **Size**: ~3.2 GB (one-time download)
- **Performance**: Tested on MacBook Air M3 16GB - runs smoothly with no issues

*Note: While GPU-optimized VLM models would be ideal, the current ONNX ecosystem has limited options. This CPU-optimized model provides an  balance of quality and performance for real-time analysis.*

