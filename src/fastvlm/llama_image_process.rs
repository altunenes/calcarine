use anyhow::Result;
use image::{DynamicImage, GenericImageView};

/// Simplified image processor for llama-cpp-2 multimodal
/// The heavy lifting is done by the mtmd module, we just need basic preprocessing
pub struct LlamaImageProcessor;

#[allow(dead_code)]
impl LlamaImageProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Basic image preprocessing - resize to reasonable dimensions
    /// The mtmd module handles the actual model-specific preprocessing
    pub fn preprocess(&self, image: &DynamicImage) -> Result<DynamicImage> {
        let (width, height) = image.dimensions();
        
        // Resize if too large (to save memory and processing time)
        let max_dimension = 1024;
        if width > max_dimension || height > max_dimension {
            let scale = if width > height {
                max_dimension as f32 / width as f32
            } else {
                max_dimension as f32 / height as f32
            };
            
            let new_width = (width as f32 * scale) as u32;
            let new_height = (height as f32 * scale) as u32;
            
            let resized = image.resize_exact(
                new_width, 
                new_height, 
                image::imageops::FilterType::Lanczos3
            );
            
            Ok(resized)
        } else {
            // Return as-is if already reasonably sized
            Ok(image.clone())
        }
    }
}

impl Default for LlamaImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}