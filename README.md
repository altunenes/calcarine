# Calcarine

Scene analysis from Apple FastVLM model 

prompt: **What is written in the upper left and upper right corners?**
3.58 seconds processing time. MacOS m3 16 GB

<div align="center">
  <img width="600" alt="calcarine" src="https://github.com/user-attachments/assets/44990581-0312-4237-80ed-623f6000794a" />
</div>



## Features

- **Multi-source input**: Images, videos, webcam feeds - We pass textures directly to the GPU
- **Real-time GPU processing**: Compute shaders with hot reload via Cuneus
- **VLM analysis**: Real-time predictions using Apple FastVLM model 
- **Interactive controls**: Real-time parameter adjustment via egui

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## What You're Actually Using

Under the hood, Calcarine is powered by [**Cuneus**](https://github.com/altunenes/cuneus). The real focus is passing textures to GPU and having the VLM make real-time predictions and analysis. The simple image effects (brightness, contrast, etc.) are just examples of what compute shaders can do.

## AI Model Details

**Model**: [Apple FastVLM-1B](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/tree/main)
- **Size**: ~1.4 GB (one-time download)
- **Storage**: Models are downloaded to `data/fastvlm/` directory
- **Performance**: Macbook M3 CPU: - ~3-4 second inference time


