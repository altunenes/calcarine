# Installation Guide

## Download

1. Go to the [Releases page](https://github.com/altunenes/calcarine/releases)
2. Download the appropriate archive for your platform:
   - **macOS**: `calcarine-x86_64-apple-darwin-coreml.tar.gz` (CoreML GPU acceleration)
   - **Windows**: `calcarine-x86_64-pc-windows-msvc-cuda.zip` (CUDA GPU + CPU fallback)
   - **Linux**: `calcarine-x86_64-unknown-linux-gnu-cuda.tar.gz` (CUDA GPU + CPU fallback)

## Prerequisites

### GStreamer Installation

**For running Calcarine (Runtime only):**
- Download and install GStreamer **Runtime** from [gstreamer.freedesktop.org](https://gstreamer.freedesktop.org/download/)

**For building from source (Development):**
- Install GStreamer **Development** package in addition to runtime

### CUDA Requirements (Windows/Linux only)

For **optimal GPU performance** on Windows and Linux:
- Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and cuDNN
- **Note**: CUDA runtime files are not included in releases due to size constraints
- **Fallback**: If CUDA is not available, the application will automatically use CPU inference

### Platform-Specific Requirements

#### Windows
- Extract the archive
- Run `calcarine.exe` or use the provided `run_calcarine.bat`

#### macOS
- Extract the archive
- **Important**: You need to give permission since the app is not signed
  - Right-click on `calcarine` → "Open" → "Open" (or use System Preferences → Security)
  - Alternative: Run `xattr -d com.apple.quarantine calcarine` in terminal
- Run `./calcarine` or use `./run_calcarine.sh`

#### Linux
- Extract the archive
- Ensure GStreamer is installed via your package manager
- Run `./calcarine` or use `./run_calcarine.sh`

## First Run

1. **Model Download**: On first launch, Calcarine will automatically download FastVLM models (~1.4 GB)
2. **Storage Location**: Models are saved to `data/fastvlm/` directory
3. **Internet Required**: Only needed for the initial model download

## Usage

- **Press 'H'** to toggle the settings UI
- **Drag & drop** media files to load them
- **Enable webcam** through the media controls
- **Adjust effects** using the real-time sliders
- **Configure AI analysis** in the dedicated panel

## Troubleshooting

### GStreamer Issues
- Ensure GStreamer is properly installed and in your PATH
- On Windows, restart after GStreamer installation
- On Linux, install GStreamer plugins: `gstreamer1.0-plugins-*`

### macOS Security
- If you get "cannot be opened because it is from an unidentified developer":
  ```bash
  xattr -d com.apple.quarantine calcarine
  ```
