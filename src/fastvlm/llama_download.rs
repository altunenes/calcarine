use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;
use reqwest::Client;
use tokio::io::AsyncWriteExt;
use futures_util::StreamExt;

/// Model file information for llama-cpp-2 multimodal models
struct ModelFile {
    name: &'static str,
    description: &'static str,
    recommended_url: &'static str,
    size_info: &'static str,
}

const RECOMMENDED_MODELS: &[ModelFile] = &[
    ModelFile {
        name: "gemma-3-4b-it-Q4_K_M.gguf",
        description: "Gemma 3 4B Instruct - Tested with llama-cpp-2 mtmd example",
        recommended_url: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        size_info: "~2.4 GB",
    },
    ModelFile {
        name: "mmproj-F16.gguf", 
        description: "Gemma 3 multimodal projection weights (required)",
        recommended_url: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf",
        size_info: "~588 MB",
    },
];

/// Alternative models compatible with llama-cpp-2
const ALTERNATIVE_MODELS: &[ModelFile] = &[
    ModelFile {
        name: "llava-v1.6-mistral-7b.Q4_K_M.gguf",
        description: "LLaVA v1.6 Mistral 7B - More capable but larger", 
        recommended_url: "https://huggingface.co/cjpais/llava-v1.6-mistral-7B-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf",
        size_info: "~4.1 GB",
    },
];

/// Check if required model files exist in the directory
pub fn check_model_files(model_dir: &Path) -> (bool, Vec<String>) {
    let mut missing_files = Vec::new();
    let mut has_model = false;
    let mut has_mmproj = false;

    // Check for any .gguf model file
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.ends_with(".gguf") && !filename.contains("mmproj") {
                    has_model = true;
                } else if filename.contains("mmproj") && filename.ends_with(".gguf") {
                    has_mmproj = true;
                }
            }
        }
    }

    if !has_model {
        missing_files.push("Main model file (.gguf)".to_string());
    }
    if !has_mmproj {
        missing_files.push("Multimodal projection file (mmproj-*.gguf)".to_string());
    }

    (has_model && has_mmproj, missing_files)
}

/// Find existing model files in directory
pub fn find_model_files(model_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let mut model_path = None;
    let mut mmproj_path = None;

    if !model_dir.exists() {
        return Err(anyhow::anyhow!("Model directory does not exist: {}", model_dir.display()));
    }

    // Scan directory for model files
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            if filename.ends_with(".gguf") {
                if filename.contains("mmproj") {
                    mmproj_path = Some(path);
                } else {
                    // Prioritize certain model types
                    if model_path.is_none() || 
                       filename.contains("llava") || 
                       filename.contains("phi-3-vision") ||
                       filename.contains("mistral") {
                        model_path = Some(path);
                    }
                }
            }
        }
    }

    match (model_path, mmproj_path) {
        (Some(model), Some(mmproj)) => {
            info!("Found model: {}", model.display());
            info!("Found mmproj: {}", mmproj.display());
            Ok((model, mmproj))
        }
        (None, _) => Err(anyhow::anyhow!("No model .gguf file found in {}", model_dir.display())),
        (_, None) => Err(anyhow::anyhow!("No mmproj .gguf file found in {}", model_dir.display())),
    }
}

/// Download a single model file with progress
async fn download_file(client: &Client, url: &str, dest_path: &Path, name: &str) -> Result<()> {
    println!("Downloading {}...", name);
    
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
        let chunk = chunk.with_context(|| "Error reading download stream")?;
        
        file.write_all(&chunk)
            .await
            .with_context(|| "Error writing to file")?;
        
        downloaded += chunk.len() as u64;
        
        if total_size > 0 && (downloaded % (50 * 1024 * 1024) == 0 || downloaded == total_size) {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            println!("   {}: {:.1}% ({:.1} MB / {:.1} MB)", 
                     name,
                     progress, 
                     downloaded as f64 / (1024.0 * 1024.0),
                     total_size as f64 / (1024.0 * 1024.0));
        }
    }

    file.flush().await.with_context(|| "Failed to flush file")?;
    println!("{} downloaded successfully", name);
    Ok(())
}

/// Download recommended models automatically
pub async fn download_models() -> Result<PathBuf> {
    let model_dir = get_default_model_dir();
    println!("Starting model download to: {}", model_dir.display());
    
    // Create directory
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("Failed to create model directory: {}", model_dir.display()))?;

    let client = Client::new();
    
    // Download main model
    let main_model = &RECOMMENDED_MODELS[0];
    let main_path = model_dir.join(main_model.name);
    if !main_path.exists() {
        download_file(&client, main_model.recommended_url, &main_path, main_model.name).await?;
    } else {
        println!("{} already exists, skipping", main_model.name);
    }
    
    // Download mmproj
    let mmproj_model = &RECOMMENDED_MODELS[1];
    let mmproj_path = model_dir.join(mmproj_model.name);
    if !mmproj_path.exists() {
        download_file(&client, mmproj_model.recommended_url, &mmproj_path, mmproj_model.name).await?;
    } else {
        println!("{} already exists, skipping", mmproj_model.name);
    }
    
    println!("Model download completed!");
    println!("Models stored at: {}", model_dir.display());
    
    Ok(model_dir)
}

/// Print instructions for manual model download
pub fn print_download_instructions() {
    println!();
    println!("MULTIMODAL MODEL SETUP REQUIRED");
    println!("=====================================");
    println!();
    println!("Calcarine now uses llama-cpp-2 with native multimodal support!");
    println!("You need to download two files:");
    println!();
    
    println!("RECOMMENDED MODELS:");
    for model in RECOMMENDED_MODELS {
        println!("  • {}", model.name);
        println!("    {} ({})", model.description, model.size_info);
        println!("    {}", model.recommended_url);
        println!();
    }
    
    println!("SETUP INSTRUCTIONS:");
    let model_dir = get_default_model_dir();
    println!("  Target directory: {}", model_dir.display());
    println!("  1. Create model directory");
    println!("  2. Download both files to this directory");
    println!("  3. Restart Calcarine");
    println!();
    
    println!("AUTOMATIC DOWNLOAD AVAILABLE:");
    println!("  The app can download models automatically when you click");
    println!("  the 'Download Models' button in the AI settings panel.");
    println!();
}

/// Get the default model directory using system-standard locations
pub fn get_default_model_dir() -> PathBuf {
    let model_dir = if cfg!(target_os = "macos") {
        // macOS: ~/Library/Caches/Calcarine/models
        dirs::cache_dir()
            .map(|cache| cache.join("Calcarine").join("models"))
            .unwrap_or_else(|| PathBuf::from("data/models"))
    } else if cfg!(target_os = "windows") {
        // Windows: %APPDATA%\Calcarine\models
        dirs::config_dir()
            .map(|config| config.join("Calcarine").join("models"))
            .unwrap_or_else(|| PathBuf::from("data/models"))
    } else {
        // Linux: ~/.local/share/calcarine/models
        dirs::data_local_dir()
            .map(|data| data.join("calcarine").join("models"))
            .unwrap_or_else(|| PathBuf::from("data/models"))
    };
    
    // Create the directory structure if it doesn't exist
    if let Err(e) = std::fs::create_dir_all(&model_dir) {
        info!("Could not create model directory {}, falling back to local: {}", model_dir.display(), e);
        // Fallback to local directory
        let fallback = PathBuf::from("data/models");
        if std::fs::create_dir_all(&fallback).is_ok() {
            fallback
        } else {
            PathBuf::from(".")
        }
    } else {
        model_dir
    }
}

/// Try to auto-detect and setup model configuration
pub fn setup_model_config() -> Result<(String, String)> {
    let model_dir = get_default_model_dir();
    
    // Check if models exist
    let (models_exist, missing_files) = check_model_files(&model_dir);
    
    if models_exist {
        // Try to find the model files
        let (model_path, mmproj_path) = find_model_files(&model_dir)?;
        return Ok((
            model_path.to_string_lossy().to_string(),
            mmproj_path.to_string_lossy().to_string(),
        ));
    }
    
    // Models don't exist, print instructions
    print_download_instructions();
    
    Err(anyhow::anyhow!(
        "Required model files not found. Missing: {}. Please download the models and restart.",
        missing_files.join(", ")
    ))
}