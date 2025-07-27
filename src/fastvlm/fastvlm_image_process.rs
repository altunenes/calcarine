use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;

/// FastVLM Image Processor following CLIP-style preprocessing
/// Based on config: 1024x1024 center crop, rescale factor 0.00392156862745098 (1/255)
pub struct FastVLMImageProcessor {
    crop_size: (u32, u32),
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    rescale_factor: f32,
}

impl FastVLMImageProcessor {
    pub fn new() -> Self {
        Self {
            crop_size: (1024, 1024),
            image_mean: vec![0.0, 0.0, 0.0],
            image_std: vec![1.0, 1.0, 1.0],
             // 1/255
            rescale_factor: 0.00392156862745098,
        }
    }

    /// Main preprocessing function for FastVLM
    pub fn preprocess(&self, image: &DynamicImage) -> Result<FastVLMBatchFeature> {
        // Convert to RGB if needed
        let rgb_image = DynamicImage::ImageRgb8(image.to_rgb8());
        
        // Resize with padding to 1024x1024 (preserves entire image content)
        let processed = self.resize_with_padding(&rgb_image, self.crop_size.0, self.crop_size.1)?;
        
        // Convert to tensor and normalize
        let pixel_values = self.to_tensor(&processed)?;
        
        Ok(FastVLMBatchFeature {
            pixel_values,
        })
    }


    fn resize_with_padding(&self, image: &DynamicImage, target_width: u32, target_height: u32) -> Result<DynamicImage> {
        let (orig_width, orig_height) = image.dimensions();
        let scale_x = target_width as f32 / orig_width as f32;
        let scale_y = target_height as f32 / orig_height as f32;
        let scale = scale_x.min(scale_y);
        
        let new_width = (orig_width as f32 * scale) as u32;
        let new_height = (orig_height as f32 * scale) as u32;
        
        let resized = image.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
        
        let mut canvas = image::RgbImage::new(target_width, target_height);
        
        let x_offset = (target_width - new_width) / 2;
        let y_offset = (target_height - new_height) / 2;
        
        let resized_rgb = resized.to_rgb8();
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = resized_rgb.get_pixel(x, y);
                canvas.put_pixel(x + x_offset, y + y_offset, *pixel);
            }
        }
        
        Ok(DynamicImage::ImageRgb8(canvas))
    }


    /// Convert image to tensor format expected by FastVLM
    /// Output shape: [1, 3, 1024, 1024] (batch, channels, height, width)
    fn to_tensor(&self, image: &DynamicImage) -> Result<Array4<f32>> {
        let (width, height) = image.dimensions();
        
        // Create tensor: [1, 3, height, width] - 4D tensor as expected by vision encoder
        let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
        
        // Fill tensor with normalized pixel values
        for (x, y, pixel) in image.pixels() {
            for c in 0..3 {
                let normalized_value = (pixel[c] as f32 * self.rescale_factor - self.image_mean[c]) / self.image_std[c];
                tensor[[0, c, y as usize, x as usize]] = normalized_value;
            }
        }
        
        Ok(tensor)
    }
}

pub struct FastVLMBatchFeature {
    pub pixel_values: Array4<f32>,
}

impl Default for FastVLMImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}