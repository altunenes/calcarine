use anyhow::{Context, Result};
use reqwest::Client;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use futures_util::StreamExt;
use tracing::{info, error};

/// FastVLM model file information
struct ModelFile {
    name: &'static str,
    url: &'static str,
    size_mb: f32,
}

const FASTVLM_MODELS: &[ModelFile] = &[
    ModelFile {
        name: "vision_encoder.onnx",
        url: "https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/resolve/main/onnx/vision_encoder.onnx",
        size_mb: 450.0,
    },
    ModelFile {
        name: "embed_tokens.onnx", 
        url: "https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/resolve/main/onnx/embed_tokens.onnx",
        size_mb: 12.0,
    },
    ModelFile {
        name: "decoder_model_merged.onnx",
        url: "https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/resolve/main/onnx/decoder_model_merged.onnx", 
        size_mb: 920.0,
    },
    ModelFile {
        name: "tokenizer.json",
        url: "https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/resolve/main/tokenizer.json",
        size_mb: 2.2,
    },
];

/// Download FastVLM models to the specified directory
pub async fn download_fastvlm_models(model_dir: &Path) -> Result<()> {
    println!("🚀 Starting FastVLM model download to: {}", model_dir.display());
    info!("Starting FastVLM model download to: {}", model_dir.display());
    
    fs::create_dir_all(model_dir)
        .with_context(|| format!("Failed to create model directory: {}", model_dir.display()))?;

    let client = Client::new();
    let total_size: f32 = FASTVLM_MODELS.iter().map(|m| m.size_mb).sum();
    
    println!("📦 Total download size: {:.1} GB ({} files)", total_size / 1024.0, FASTVLM_MODELS.len());
    info!("Total download size: {:.1} GB", total_size / 1024.0);

    for (index, model) in FASTVLM_MODELS.iter().enumerate() {
        let file_path = model_dir.join(model.name);
        
        // Skip if file already exists
        if file_path.exists() {
            println!("✅ Model {} already exists, skipping", model.name);
            info!("Model {} already exists, skipping", model.name);
            continue;
        }

        println!("📥 [{}/{}] Downloading {} ({:.1} MB)...", 
                 index + 1, FASTVLM_MODELS.len(), model.name, model.size_mb);
        info!("Downloading {} ({:.1} MB)...", model.name, model.size_mb);
        
        match download_file(&client, model.url, &file_path).await {
            Ok(_) => {
                println!("✅ [{}/{}] Successfully downloaded {}", 
                         index + 1, FASTVLM_MODELS.len(), model.name);
                info!("Successfully downloaded {}", model.name);
            },
            Err(e) => {
                println!("❌ Failed to download {}: {}", model.name, e);
                error!("Failed to download {}: {}", model.name, e);
                return Err(e);
            }
        }
    }

    println!("🎉 FastVLM model download completed successfully!");
    println!("");
    println!("📍 Models stored at: {}", model_dir.display());
    println!("💾 Total space used: ~{:.1} GB", total_size / 1024.0);
    println!("");
    println!("🗑️  To remove models later, delete this folder:");
    println!("   {}", model_dir.display());
    println!("");
    
    info!("FastVLM model download completed successfully");
    Ok(())
}

async fn download_file(client: &Client, url: &str, dest_path: &Path) -> Result<()> {
    let response = client.get(url)
        .send()
        .await
        .with_context(|| format!("Failed to request {}", url))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("Download failed with status: {}", response.status()));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    
    let mut file = tokio::fs::File::create(dest_path)
        .await
        .with_context(|| format!("Failed to create file: {}", dest_path.display()))?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .with_context(|| "Error reading download stream")?;
        
        file.write_all(&chunk)
            .await
            .with_context(|| "Error writing to file")?;
        
        downloaded += chunk.len() as u64;
        
        if total_size > 0 && (downloaded % (10 * 1024 * 1024) == 0 || downloaded == total_size) {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            println!("   📊 Progress: {:.1}% ({:.1} MB / {:.1} MB)", 
                     progress, 
                     downloaded as f64 / (1024.0 * 1024.0),
                     total_size as f64 / (1024.0 * 1024.0));
            info!("Download progress: {:.1}% ({:.1} MB / {:.1} MB)", 
                  progress, 
                  downloaded as f64 / (1024.0 * 1024.0),
                  total_size as f64 / (1024.0 * 1024.0));
        }
    }

    file.flush().await
        .with_context(|| "Failed to flush file")?;

    Ok(())
}


/// Get the default model directory using system-standard locations
pub fn get_default_model_dir() -> PathBuf {
    // Use platform-specific standard directories
    let model_dir = if cfg!(target_os = "macos") {
        // macOS: ~/Library/Caches/Calcarine/models
        dirs::cache_dir()
            .map(|cache| cache.join("Calcarine").join("models"))
            .unwrap_or_else(|| PathBuf::from("data/fastvlm"))
    } else if cfg!(target_os = "windows") {
        // Windows: %APPDATA%\Calcarine\models
        dirs::config_dir()
            .map(|config| config.join("Calcarine").join("models"))
            .unwrap_or_else(|| PathBuf::from("data/fastvlm"))
    } else {
        // Linux: ~/.local/share/calcarine/models
        dirs::data_local_dir()
            .map(|data| data.join("calcarine").join("models"))
            .unwrap_or_else(|| PathBuf::from("data/fastvlm"))
    };
    
    // Create the directory structure if it doesn't exist
    if let Err(e) = std::fs::create_dir_all(&model_dir) {
        info!("Could not create model directory {}, falling back to local: {}", model_dir.display(), e);
        // Fallback to local directory
        let fallback = PathBuf::from("data/fastvlm");
        if let Err(_) = std::fs::create_dir_all(&fallback) {
            PathBuf::from(".")
        } else {
            fallback
        }
    } else {
        model_dir
    }
}