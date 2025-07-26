use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array2, Array3, Array4, s};
use ort::{
    session::Session,
    session::builder::GraphOptimizationLevel,
    value::{Tensor, TensorRef}
};

#[cfg(target_os = "macos")]
use ort::{
    execution_providers::CoreMLExecutionProvider,
    execution_providers::coreml::CoreMLComputeUnits,
};

#[cfg(any(target_os = "windows", target_os = "linux"))]
use ort::{
    execution_providers::CUDAExecutionProvider,
    execution_providers::CPUExecutionProvider,
};
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

use super::fastvlm_image_process::FastVLMImageProcessor;

// FastVLM special tokens
const EOS_TOKEN_ID: i64 = 151645; // <|im_end|>
const IM_END_TOKEN_ID: i64 = 151645; // <|im_end|>
const IMAGE_TOKEN_ID: i64 = 151646; // <image>

#[derive(Debug, Clone)]
pub struct FastVLMAnalysisResult {
    pub text: String,
    pub timestamp: Instant,
    pub processing_time: Duration,
}

#[derive(Debug, Clone)]
pub struct FastVLMConfig {
    pub max_response_length: usize,
    pub default_prompt: String,
}

impl Default for FastVLMConfig {
    fn default() -> Self {
        Self {
            max_response_length: 30,
            default_prompt: "Describe this image briefly.".to_string(),
        }
    }
}

pub struct FastVLM {
    tokenizer: Tokenizer,
    vision_encoder: Session,
    embed_tokens: Session,
    decoder: Session,
    config: FastVLMConfig,
    image_processor: FastVLMImageProcessor,
    
    last_analysis: Option<FastVLMAnalysisResult>,
    last_analysis_time: Instant,
}

impl FastVLM {
    pub async fn new(data_dir: &Path, config: FastVLMConfig) -> Result<Self> {
        tracing::info!("Initializing FastVLM with CoreML GPU acceleration...");
        
        let _ = ort::init()
            .with_name("fastvlm")
            .commit()
            .map_err(|e| {
                tracing::debug!("ONNX Runtime already initialized or failed: {:?}", e);
            });
        
        let tokenizer_path = data_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Error loading tokenizer: {:?}", e))?;
        

        let create_session = |model_path: &str| -> Result<Session> {
            let mut builder = Session::builder()
                .map_err(|e| anyhow::anyhow!("Session builder error: {:?}", e))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("Optimization level error: {:?}", e))?;

            // Platform-specific execution providers
            #[cfg(target_os = "macos")]
            {
                builder = builder.with_execution_providers([
                    CoreMLExecutionProvider::default()
                        .with_compute_units(CoreMLComputeUnits::CPUAndGPU)
                        .with_static_input_shapes(true)
                        .build()
                ])
                .map_err(|e| anyhow::anyhow!("CoreML execution provider error: {:?}", e))?;
                
                tracing::info!("Using CoreML execution provider for {}", model_path);
            }

            #[cfg(any(target_os = "windows", target_os = "linux"))]
            {
                builder = builder.with_execution_providers([
                    CUDAExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build()
                ])
                .map_err(|e| anyhow::anyhow!("CUDA/CPU execution provider error: {:?}", e))?;
                
                tracing::info!("Using CUDA + CPU execution providers for {}", model_path);
            }

            Ok(builder.commit_from_file(data_dir.join(model_path))
                .map_err(|e| anyhow::anyhow!("Model loading error for {}: {:?}", model_path, e))?)
        };
        
        // Load models
        let vision_encoder = create_session("vision_encoder.onnx")?;
        let embed_tokens = create_session("embed_tokens.onnx")?;
        let decoder = create_session("decoder_model_merged.onnx")?;
        
        tracing::info!("FastVLM models loaded successfully");
        
        Ok(Self {
            tokenizer,
            vision_encoder,
            embed_tokens,
            decoder,
            config,
            image_processor: FastVLMImageProcessor::new(),
            last_analysis: None,
            last_analysis_time: Instant::now(),
        })
    }
    
    pub fn analyze_frame_sync(
        &mut self,
        image_data: Vec<u8>,
        width: u32,
        height: u32,
        prompt: Option<String>,
    ) -> Result<FastVLMAnalysisResult> {
        let start_time = Instant::now();
        let prompt = prompt.unwrap_or_else(|| self.config.default_prompt.clone());
        
        tracing::debug!("Starting FastVLM analysis for {}x{} image", width, height);
        
        let image = self.rgba_to_dynamic_image(image_data, width, height)?;
        
        let generated_text = self.generate_text_sync(&image, &prompt)?;
        
        let processing_time = start_time.elapsed();
        let result = FastVLMAnalysisResult {
            text: generated_text,
            timestamp: start_time,
            processing_time,
        };
        
        self.last_analysis = Some(result.clone());
        self.last_analysis_time = start_time;
        
        tracing::info!("FastVLM analysis completed in {:?}: {}", processing_time, result.text);
        
        Ok(result)
    }
    
    fn rgba_to_dynamic_image(&self, data: Vec<u8>, width: u32, height: u32) -> Result<DynamicImage> {
        let expected_size = (width * height * 4) as usize;
        
        if data.len() != expected_size {
            tracing::warn!("Image data size mismatch: got {}, expected {}. Attempting to handle padding.", 
                         data.len(), expected_size);
            
            if data.len() > expected_size {
                let unpadded_data = data.into_iter().take(expected_size).collect();
                return self.rgba_to_dynamic_image(unpadded_data, width, height);
            } else {
                return Err(anyhow::anyhow!(
                    "Image data length {} is less than expected size {}",
                    data.len(),
                    expected_size
                ));
            }
        }
        
        let image_buffer = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;
        
        Ok(DynamicImage::ImageRgba8(image_buffer))
    }
    
    fn generate_text_sync(&mut self, image: &DynamicImage, text: &str) -> Result<String> {
        tracing::debug!("Processing image and generating text response");
        
        // Process image using FastVLM image processor
        let image_features = self.get_image_features(image)?;
        tracing::debug!("Image features extracted with shape: {:?}", image_features.shape());
        
        // Format prompt using FastVLM chat template
        let formatted_prompt = self.format_chat_template(text);
        tracing::debug!("Formatted prompt length: {} chars", formatted_prompt.len());
        
        // Tokenize the prompt
        let encoding = self.tokenizer.encode(formatted_prompt, true)
            .map_err(|e| anyhow::anyhow!("Error encoding: {:?}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        tracing::debug!("Token IDs length: {}", input_ids.len());
        
        let image_token_position = input_ids.iter().position(|&id| id == IMAGE_TOKEN_ID)
            .unwrap_or(input_ids.len() / 2);
        
        tracing::debug!("Image token position: {}", image_token_position);
        
        let input_embeds = self.get_token_embeddings(&input_ids)?;
        
        let fused_embeds = self.fuse_image_text_embeddings(&input_embeds, &image_features, image_token_position)?;
        
        let generated_text = self.generate_with_decoder(&fused_embeds)?;
        
        Ok(generated_text.trim().to_string())
    }
    
    fn get_image_features(&mut self, image: &DynamicImage) -> Result<Array3<f32>> {
        let batch_feature = self.image_processor.preprocess(image)?;
        
        tracing::debug!("Vision encoder input shape: {:?}", batch_feature.pixel_values.shape());
        
        let outputs = self.vision_encoder.run(ort::inputs![
            "pixel_values" => Tensor::from_array(batch_feature.pixel_values)?,
        ])?;
        
        let output_name = if outputs.contains_key("last_hidden_state") {
            "last_hidden_state"
        } else if outputs.contains_key("image_features") {
            "image_features"
        } else if outputs.contains_key("output") {
            "output"
        } else {
            outputs.keys().next().unwrap()
        };
        
        let image_features_view = outputs[output_name].try_extract_array::<f32>()?;
        tracing::debug!("Vision encoder output shape: {:?}", image_features_view.shape());
        
        // The vision encoder already outputs in sequence format [batch, seq_len, hidden_dim]
        let image_features = match image_features_view.ndim() {
            2 => {
                // [seq_len, hidden_dim] -> [1, seq_len, hidden_dim] 
                let shape = image_features_view.shape();
                image_features_view.to_shape((1, shape[0], shape[1]))?.to_owned()
            },
            3 => {
                // Already [batch, seq_len, hidden_dim]
                image_features_view.into_dimensionality::<ndarray::Ix3>()?.to_owned()
            },
            _ => {
                return Err(anyhow::anyhow!("Unexpected vision encoder output dimensionality: {}", image_features_view.ndim()));
            }
        };
        
        tracing::debug!("Final image features shape: {:?}", image_features.shape());
        Ok(image_features)
    }
    
    fn get_token_embeddings(&mut self, input_ids: &[i64]) -> Result<Array3<f32>> {
        let input_ids_array = Array2::from_shape_vec((1, input_ids.len()), input_ids.to_vec())?;
        
        let outputs = self.embed_tokens.run(ort::inputs![
            "input_ids" => TensorRef::from_array_view(&input_ids_array)?,
        ])?;
        
        let embeddings_view = outputs["inputs_embeds"].try_extract_array::<f32>()?;
        let embeddings = embeddings_view.into_dimensionality::<ndarray::Ix3>()?.to_owned();
        
        tracing::debug!("Token embeddings shape: {:?}", embeddings.shape());
        Ok(embeddings)
    }
    
    fn fuse_image_text_embeddings(&self, text_embeds: &Array3<f32>, image_features: &Array3<f32>, image_token_pos: usize) -> Result<Array3<f32>> {
        let text_seq_len = text_embeds.shape()[1];
        let image_seq_len = image_features.shape()[1]; 
        let hidden_dim = text_embeds.shape()[2];
        
        tracing::debug!("Fusing embeddings - text seq: {}, image seq: {}, token pos: {}", 
                       text_seq_len, image_seq_len, image_token_pos);
        
        // Image features are already in the right format [1, seq_len, hidden_dim]
        // Just need to ensure dimensions match
        if image_features.shape()[2] != hidden_dim {
            return Err(anyhow::anyhow!("Image feature dimension {} doesn't match text dimension {}", 
                                     image_features.shape()[2], hidden_dim));
        }
        
        // Take a reasonable number of image tokens
        let max_image_tokens = 256; 
        let actual_image_tokens = image_seq_len.min(max_image_tokens);
        let final_image_embeds = image_features.slice(s![.., ..actual_image_tokens, ..]).to_owned();
        
        // Create fused embedding by inserting image features at the image token position
        let total_seq_len = text_seq_len + actual_image_tokens;
        let mut fused_embeds = Array3::<f32>::zeros((1, total_seq_len, hidden_dim));
        
        // Copy text embeddings before image position
        if image_token_pos > 0 {
            fused_embeds.slice_mut(s![.., ..image_token_pos, ..])
                .assign(&text_embeds.slice(s![.., ..image_token_pos, ..]));
        }
        
        // Insert image embeddings at the correct position
        fused_embeds.slice_mut(s![.., image_token_pos..image_token_pos + actual_image_tokens, ..])
            .assign(&final_image_embeds);
        
        // Copy remaining text embeddings after image
        if image_token_pos < text_seq_len {
            fused_embeds.slice_mut(s![.., image_token_pos + actual_image_tokens.., ..])
                .assign(&text_embeds.slice(s![.., image_token_pos.., ..]));
        }
        
        tracing::debug!("Fused embeddings shape: {:?} (text: {}, image: {})", 
                       fused_embeds.shape(), text_seq_len, actual_image_tokens);
        
        Ok(fused_embeds)
    }

    fn generate_with_decoder(&mut self, input_embeds: &Array3<f32>) -> Result<String> {
        tracing::debug!("Starting decoder generation with input embeds shape: {:?}", input_embeds.shape());
        
        // Create position_ids for the sequence
        let seq_len = input_embeds.shape()[1];
        let position_ids: Array2<i64> = Array2::from_shape_fn((1, seq_len), |(_, i)| i as i64);
        
        // Create attention mask (all ones for now)
        let attention_mask: Array2<i64> = Array2::ones((1, seq_len));
        
        // Create empty past key-value cache for first inference
        // Based on the error, the model has at least 13 layers (layer.12 exists)
        let num_layers = 24; // Standard for this model size
        let num_kv_heads = 2;
        let head_dim = 64; // 896 / 14
        
        let empty_kv_tensors: Vec<Array4<f32>> = (0..num_layers * 2)
            .map(|_| Array4::<f32>::zeros((1, num_kv_heads, 0, head_dim)))
            .collect();
        
        // Autoregressive generation loop
        let mut generated_tokens = Vec::with_capacity(self.config.max_response_length);
        let mut current_inputs_embeds = input_embeds.clone();
        let mut current_attention_mask = attention_mask;
        let mut current_position_ids = position_ids;
        let mut past_key_values = empty_kv_tensors;
        
        for step in 0..self.config.max_response_length {
            // Prepare inputs for current step
            let mut model_inputs = ort::inputs![
                "inputs_embeds" => TensorRef::from_array_view(&current_inputs_embeds)?,
                "position_ids" => TensorRef::from_array_view(&current_position_ids)?,
                "attention_mask" => TensorRef::from_array_view(&current_attention_mask)?,
            ];
            
            // Add past key-value pairs
            for i in 0..num_layers {
                model_inputs.push((
                    format!("past_key_values.{}.key", i).into(),
                    TensorRef::from_array_view(&past_key_values[i * 2])?.into()
                ));
                model_inputs.push((
                    format!("past_key_values.{}.value", i).into(),
                    TensorRef::from_array_view(&past_key_values[i * 2 + 1])?.into()
                ));
            }
            
            // Run decoder
            let outputs = self.decoder.run(model_inputs)?;
            
            let logits = outputs["logits"].try_extract_array::<f32>()?
                .into_dimensionality::<ndarray::Ix3>()?
                .to_owned();
            
            // Update past key-values for next iteration
            let mut new_past_key_values = Vec::with_capacity(num_layers * 2);
            for i in 0..num_layers {
                let key = outputs[format!("present.{}.key", i)]
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<ndarray::Ix4>()?
                    .to_owned();
                let value = outputs[format!("present.{}.value", i)]
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<ndarray::Ix4>()?
                    .to_owned();
                new_past_key_values.push(key);
                new_past_key_values.push(value);
            }
            
            // release the borrow
            drop(outputs);
            
            // Get next token from the last position with temperature sampling
            let seq_len = logits.shape()[1];
            let vocab_size = logits.shape()[2].min(151646);
            let last_token_logits = logits.slice(s![0, seq_len-1, ..vocab_size]);
            let next_token_id = self.sample_token(&last_token_logits, 0.7)?;
            
            // Check for end tokens
            if next_token_id == EOS_TOKEN_ID || next_token_id == IM_END_TOKEN_ID {
                tracing::debug!("End token detected, stopping generation at step {}", step + 1);
                break;
            }
            
            generated_tokens.push(next_token_id as u32);
            
            // Update the past key values
            past_key_values = new_past_key_values;
            
            // Prepare inputs for next step
            current_inputs_embeds = self.get_token_embeddings(&[next_token_id])?;
            
            // Update attention mask and position IDs
            let current_seq_len = current_attention_mask.shape()[1];
            current_attention_mask = Array2::ones((1, current_seq_len + 1));
            current_position_ids = Array2::from_shape_fn((1, 1), |(_, _)| current_seq_len as i64);
        }
        
        tracing::debug!("Generated {} tokens total", generated_tokens.len());
        
        if generated_tokens.is_empty() {
            Ok("No response generated.".to_string())
        } else {
            // Decode all generated tokens to text
            let generated_text = self.tokenizer.decode(&generated_tokens, true)
                .map_err(|e| anyhow::anyhow!("Decode error: {:?}", e))?;
            tracing::debug!("Decoded text: '{}'", generated_text);
            Ok(generated_text.trim().to_string())
        }
    }
    
    fn sample_token(&self, logits: &ndarray::ArrayView1<f32>, temperature: f32) -> Result<i64> {
        // For better performance
        const TOP_K: usize = 50;
        
        // Get top-k tokens with their indices
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &logit)| (i, logit))
            .collect();
        
        // Partial sort to get top-k elements
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_logits.truncate(TOP_K);
        
        // Apply temperature and softmax to top-k
        let max_logit = indexed_logits[0].1;
        let mut probabilities: Vec<(usize, f32)> = indexed_logits.into_iter()
            .map(|(idx, logit)| {
                let scaled = (logit - max_logit) / temperature;
                (idx, scaled.exp())
            })
            .collect();
        
        let sum_exp: f32 = probabilities.iter().map(|(_, prob)| prob).sum();
        for (_, prob) in probabilities.iter_mut() {
            *prob /= sum_exp;
        }
        
        // Simple random sampling
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let random_val = ((seed as u64).wrapping_mul(1103515245).wrapping_add(12345) % 1000000) as f32 / 1000000.0;
        
        // Sample from top-k distribution
        let mut cumulative_prob = 0.0;
        for (idx, prob) in probabilities {
            cumulative_prob += prob;
            if random_val <= cumulative_prob {
                return Ok(idx as i64);
            }
        }
        
        // Fallback to most likely token
        Ok(0)
    }

    fn format_chat_template(&self, text: &str) -> String {
        // FastVLM chat template image token
        format!("<|im_start|>system\nYou are a helpful vision assistant that describes images accurately.<|im_end|>\n<|im_start|>user\n<image>\n{}<|im_end|>\n<|im_start|>assistant\n", text)
    }
    
}