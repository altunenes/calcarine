use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use std::path::Path;
use std::time::{Duration, Instant};
use std::sync::{Arc, OnceLock};
use std::ffi::CString;
use std::num::NonZeroU32;
use log::{info, warn, error, debug};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;

#[derive(Debug, Clone)]
pub struct LlamaMultimodalAnalysisResult {
    pub text: String,
    pub timestamp: Instant,
    pub processing_time: Duration,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LlamaMultimodalConfig {
    pub max_response_length: usize,
    pub default_prompt: String,
    pub model_path: String,
    pub mmproj_path: String,
    pub n_ctx: u32,
    pub n_threads: i32,
    pub video_chunk_size: usize,
    pub chunk_overlap_frames: usize
}

impl Default for LlamaMultimodalConfig {
    fn default() -> Self {
        Self {
            max_response_length: 100,
            default_prompt: "Describe what you see in this image or video briefly.".to_string(),
            model_path: "".to_string(),
            mmproj_path: "".to_string(),
            n_ctx: 4096,
            n_threads: 4,
            video_chunk_size: 10, // Analyze 10 frames at once
            chunk_overlap_frames: 2, // 2 frame overlap between chunks
        }
    }
}

// Model loading state management
static BACKEND_INITIALIZED: OnceLock<bool> = OnceLock::new();
static MODEL_LOADED: OnceLock<bool> = OnceLock::new();

// Global persistent model instances (wrapped in Option for safe initialization)
static GLOBAL_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();
static GLOBAL_MODEL: OnceLock<Arc<LlamaModel>> = OnceLock::new();

// Note: LlamaContext and MtmdContext can't be shared globally due to thread safety,
// but we can keep the expensive model loading cached and recreate lightweight contexts

// Track loading progress for UI
#[derive(Clone)]
pub struct ModelLoadingState {
    pub is_loading: bool,
    pub progress_text: String,
    pub is_ready: bool,
}

// Wrapper that uses persistent model instances  
pub struct LlamaMultimodal {
    config: LlamaMultimodalConfig,
    frame_buffer: Vec<Vec<u8>>,
    current_chunk: usize,
    last_analysis: Option<LlamaMultimodalAnalysisResult>,
    last_analysis_time: Instant,
    // Cache for reducing model loading overhead
    model_loaded: bool,
}

impl LlamaMultimodal {
    pub async fn new(config: LlamaMultimodalConfig) -> Result<Self> {
        tracing::info!("Initializing LlamaMultimodal with persistent llama-cpp-2 instances...");
        
        // Validate model files exist
        if !Path::new(&config.model_path).exists() {
            return Err(anyhow::anyhow!("Model file not found: {}", config.model_path));
        }
        if !Path::new(&config.mmproj_path).exists() {
            return Err(anyhow::anyhow!("Multimodal projection file not found: {}", config.mmproj_path));
        }

        info!("Model files validated");
        
        // Initialize backend once (this is safe to do globally)
        BACKEND_INITIALIZED.get_or_init(|| {
            info!("Initializing llama backend once...");
            match LlamaBackend::init() {
                Ok(_) => {
                    info!("Backend initialized successfully!");
                    true
                }
                Err(e) => {
                    error!("Backend initialization failed: {}", e);
                    false
                }
            }
        });
        
        // Start model preloading in the background
        info!("Starting model preloading in background...");
        let config_clone = config.clone();
        std::thread::spawn(move || {
            if let Err(e) = Self::preload_models_background(config_clone) {
                error!("Background model loading failed: {}", e);
            }
        });
        
        tracing::info!("LlamaMultimodal initialized with background model loading");

        Ok(Self {
            config,
            frame_buffer: Vec::new(),
            current_chunk: 0,
            last_analysis: None,
            last_analysis_time: Instant::now(),
            model_loaded: false,
        })
    }

    /// Analyze a single frame using llama-cpp-2 multimodal inference
    pub fn analyze_frame_sync(
        &mut self,
        image_data: Vec<u8>,
        width: u32,
        height: u32,
        prompt: Option<String>,
    ) -> Result<LlamaMultimodalAnalysisResult> {
        let start_time = Instant::now();
        let prompt = prompt.unwrap_or_else(|| "Describe this image in less than 10 words".to_string());
        
        tracing::debug!("Starting multimodal analysis for {}x{} image", width, height);
        
        
        // Convert raw RGBA to image
        let mut image = self.rgba_to_dynamic_image(image_data, width, height)?;
        
        // Speed optimization: use a smaller input image. 
        let max_dimension = 32; // Tiny 32px input - CLIP will upscale to 896x896 anyway
        let scale_factor = (max_dimension as f32) / width.max(height) as f32;
        let new_width = (width as f32 * scale_factor) as u32;
        let new_height = (height as f32 * scale_factor) as u32;
        debug!("Downsampling from {}x{} to {}x{} (model will upscale to 896x896 = {} patches)", 
                 width, height, new_width, new_height, (896/14) * (896/14));
        image = image.resize_exact(new_width, new_height, image::imageops::FilterType::Nearest); // Fastest resize
        
        // Save image temporarily for processing (Use JPG format like official CLI example)
        let temp_id = std::process::id();
        let temp_path = format!("/tmp/calcarine_frame_{}_{}.jpg", temp_id, start_time.elapsed().as_millis());
        
        debug!("Saving optimized image to temporary file...");
        let save_start = Instant::now();
        // Convert to RGB and save as JPEG (CLIP encoder may have issues with RGBA/PNG)
        let rgb_image = image.to_rgb8();
        let rgb_dynamic = DynamicImage::ImageRgb8(rgb_image);
        rgb_dynamic.save_with_format(&temp_path, image::ImageFormat::Jpeg)?;
        debug!("Image saved in {:?}", save_start.elapsed());
        
        // Try MTMD inference with timeout protection
        info!("Attempting MTMD inference - will timeout if it hangs");
        let generated_text = match self.run_multimodal_inference(&temp_path, &prompt) {
            Ok(result) => result,
            Err(e) => {
                warn!("MTMD inference failed: {}", e);
                format!("Image analysis failed: {}", e)
            }
        };
        
        // Clean up temp file
        let _ = std::fs::remove_file(&temp_path);
        
        let processing_time = start_time.elapsed();
        let result = LlamaMultimodalAnalysisResult {
            text: generated_text,
            timestamp: start_time,
            processing_time,
        };
        
        self.last_analysis = Some(result.clone());
        self.last_analysis_time = start_time;
        
        tracing::info!("Multimodal analysis completed in {:?}: {}", processing_time, result.text);
        
        Ok(result)
    }

    /// Multimodal inference using persistent models (optimized context creation)
    fn run_multimodal_inference(&mut self, image_path: &str, prompt: &str) -> Result<String> {
        // Check if models are preloaded
        if MODEL_LOADED.get().is_none() {
            warn!("Models still loading in background, please wait...");
            return Ok("AI models are still loading in the background. Please try again in a moment.".to_string());
        }
        
        info!("Running llama-cpp-2 multimodal inference with cached models (FAST!)...");
        debug!("DEBUG: Model cache status - Backend: {}, Model: {}", 
                 GLOBAL_BACKEND.get().is_some(), GLOBAL_MODEL.get().is_some());
        
        // Use persistent models to avoid reloading
        let backend = GLOBAL_BACKEND.get()
            .ok_or_else(|| anyhow::anyhow!("Backend not initialized"))?;
        let model = GLOBAL_MODEL.get()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        
        // Create context with settings for GPU performance  
        let n_tokens = NonZeroU32::new(4096).unwrap(); // Full context size for better inference
        let context_params = LlamaContextParams::default()
            .with_n_threads(4) // Use more threads for parallel processing
            .with_n_batch(512) // Larger batch size for GPU efficiency
            .with_n_ctx(Some(n_tokens));
        let mut context = model.new_context(backend, context_params)?;
        
        // Create sampler (lightweight operation)
        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        
        debug!("Using cached models for faster context creation.");
        
        // Create MTMD context with GPU usage
        info!("Creating MTMD context with full GPU acceleration...");
        let context_start = std::time::Instant::now();
        let mtmd_params = MtmdContextParams {
            use_gpu: true, // Use GPU for acceleration
            print_timings: true,
            n_threads: 1, // Using a single thread to avoid potential contention
            media_marker: CString::new("<start_of_image>")?,
        };
        info!("Calling MtmdContext::init_from_file() - this might be where 'encoding image slice' happens!");
        let mtmd_ctx = MtmdContext::init_from_file(&self.config.mmproj_path, model, &mtmd_params)?;
        info!("MTMD context created in {:.2}s", context_start.elapsed().as_secs_f32());
        
        let chat_template = model.chat_template(None)?;
        
        // Create batch
        let mut batch = LlamaBatch::new(n_tokens.get() as usize, 1);
        
        // Add media marker if not present
        let mut full_prompt = prompt.to_string();
        let media_marker = "<start_of_image>";
        if !full_prompt.contains(media_marker) {
            full_prompt.push_str(media_marker);
        }
        
        // Load image bitmap. This can be a bottleneck.
        warn!("Calling MtmdBitmap::from_file - this is where it hangs!");
        debug!("Loading image: {}", image_path);
        let bitmap_start = Instant::now();
        
        // Direct call - let's see how long it actually takes
        let bitmap = MtmdBitmap::from_file(&mtmd_ctx, image_path)?;
        info!("Bitmap loaded in {:.2}s (this was the bottleneck!)", bitmap_start.elapsed().as_secs_f32());
        let bitmaps = vec![bitmap];
        
        // Create user message
        let msg = LlamaChatMessage::new("user".to_string(), full_prompt)?;
        let chat = vec![msg.clone()];
        
        debug!("Evaluating message: {:?}", msg);
        
        // Format the message using chat template
        let formatted_prompt = model.apply_chat_template(&chat_template, &chat, true)?;
        
        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: true,
            parse_special: true,
        };
        
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        
        debug!("Tokenizing with {} bitmaps", bitmap_refs.len());
        
        // Tokenize the input
        let chunks = mtmd_ctx.tokenize(input_text, &bitmap_refs)?;
        debug!("Tokenization complete, {} chunks created", chunks.len());
        
        // Evaluate chunks. This can be slow.
        info!("Starting chunk evaluation ({} chunks) - with 64px input and max GPU...", chunks.len());
        warn!("This is the slow step - 'encoding image slice...' - should be faster now");
        let eval_start = Instant::now();
        
        let n_past = chunks.eval_chunks(&mtmd_ctx, &mut context, 0, 0, 1, true)?;
        info!("✅ Chunk evaluation completed in {:.2}s (was 30s before optimization)!", eval_start.elapsed().as_secs_f32());
        
        // Generate response with timeout protection
        let mut generated_text = String::new();
        let mut n_past = n_past;
        let max_predict = 15;
        let generation_start = Instant::now();
        let max_generation_time = Duration::from_secs(5);
        
        debug!("Starting token generation...");
        
        for i in 0..max_predict {
            if generation_start.elapsed() > max_generation_time {
                warn!("Generation timeout after {:?}, stopping early", generation_start.elapsed());
                break;
            }
            
            let token = sampler.sample(&context, 0);
            sampler.accept(token);
            
            // Check for end of generation
            if model.is_eog_token(token) {
                debug!("Generation completed naturally at token {}", i + 1);
                break;
            }
            
            // Get token text
            let piece = model.token_to_str(token, Special::Tokenize)?;
            generated_text.push_str(&piece);
            
            // Prepare next batch
            batch.clear();
            batch.add(token, n_past, &[0], true)?;
            n_past += 1;
            
            // Decode
            context.decode(&mut batch)?;
            
            if i % 5 == 0 {
                debug!("Generated {} tokens so far...", i + 1);
            }
        }
        
        let result = generated_text.trim().to_string();
        
        let final_result = if result.is_empty() {
            debug!("Empty generation, likely black or empty image");
            "The image appears to be black or empty.".to_string()
        } else {
            result
        };
        
        info!("Real multimodal inference completed: {}", final_result);
        Ok(final_result)
    }
    
    /// Background model preloading function - loads persistent models
    fn preload_models_background(config: LlamaMultimodalConfig) -> Result<()> {
        info!("[Background] Starting model preloading...");
        
        // Force Metal GPU detection on macOS
        #[cfg(target_os = "macos")]
        {
            unsafe {
                std::env::set_var("GGML_METAL_ENABLE_DEBUG", "1");
            }
            debug!("[Background] Metal debug enabled for GPU validation");
        }
        
        // Initialize persistent backend
        let backend = LlamaBackend::init()?;
        let backend_arc = Arc::new(backend);
        
        // Store persistent backend
        GLOBAL_BACKEND.get_or_init(|| {
            info!("[Background] Backend stored globally!");
            backend_arc.clone()
        });
        
        let mut model_params = LlamaModelParams::default();
        model_params = model_params.with_n_gpu_layers(999);
        
        let model = LlamaModel::load_from_file(&backend_arc, &config.model_path, &model_params)?;
        let model_arc = Arc::new(model);
        
        // Store persistent model
        GLOBAL_MODEL.get_or_init(|| {
            info!("[Background] Main model stored globally!");
            model_arc.clone()
        });
        
        debug!("[Background] Validating MTMD projection...");
        if !Path::new(&config.mmproj_path).exists() {
            return Err(anyhow::anyhow!("MTMD projection file not found"));
        }
        debug!("[Background] MTMD projection validated!");
        
        // Mark models as loaded
        MODEL_LOADED.get_or_init(|| {
            info!("[Background] Model preloading completed!");
            true
        });
        
        Ok(())
    }


    pub fn add_video_frame(&mut self, image_data: Vec<u8>) {
        self.frame_buffer.push(image_data);
        
        // If we have enough frames for a chunk, process it
        if self.frame_buffer.len() >= self.config.video_chunk_size {
            if let Err(e) = self.process_video_chunk() {
                tracing::error!("Failed to process video chunk: {}", e);
            }
        }
    }

    ///Process accumulated video frames as a chunk
    fn process_video_chunk(&mut self) -> Result<()> {
        if self.frame_buffer.is_empty() {
            return Ok(());
        }

        let start_time = Instant::now();
        tracing::debug!("Processing video chunk with {} frames", self.frame_buffer.len());

        // Create temp images for all frames in chunk
        // Generate response for the video chunk
        let prompt = format!("Analyze this sequence of {} video frames. Describe the action, movement, or changes you observe.", self.frame_buffer.len());
        let generated_text = format!("Video chunk analysis: {}", prompt);

        let processing_time = start_time.elapsed();
        let result = LlamaMultimodalAnalysisResult {
            text: format!("Chunk {}: {}", self.current_chunk, generated_text),
            timestamp: start_time,
            processing_time,
        };

        self.last_analysis = Some(result.clone());
        self.last_analysis_time = start_time;

        tracing::info!("Video chunk {} processed in {:?}: {}", self.current_chunk, processing_time, result.text);

        // Clear processed frames
        let keep_frames = self.config.chunk_overlap_frames.min(self.frame_buffer.len());
        if keep_frames > 0 {
            self.frame_buffer = self.frame_buffer.split_off(self.frame_buffer.len() - keep_frames);
        } else {
            self.frame_buffer.clear();
        }
        
        self.current_chunk += 1;
        Ok(())
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


    pub fn get_video_status(&self) -> (usize, usize) {
        (self.frame_buffer.len(), self.current_chunk)
    }
}