use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgba};
use ndarray::{Array2, Array3, Array4, s};
use ort::{
    session::Session,
    session::builder::GraphOptimizationLevel,
    execution_providers::CPUExecutionProvider,
    value::{Tensor, TensorRef}
};
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokenizers::Tokenizer;

pub mod image_process;

const EOS_TOKEN_ID: i64 = 32007;
const USER_TOKEN_ID: i64 = 32010;
const VOCAB_SIZE: usize = 32064;
#[derive(Debug, Clone)]
pub struct VisionAnalysisResult {
    pub text: String,
    pub timestamp: Instant,
    pub processing_time: Duration,
}

#[derive(Debug, Clone)]
pub struct VisionConfig {
    #[allow(dead_code)]
    pub analysis_interval: Duration,
    pub max_response_length: usize,
    pub default_prompt: String,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            analysis_interval: Duration::from_secs(8),
            max_response_length: 25,
            default_prompt: "Describe this image in 10 words or less.".to_string(),
        }
    }
}

pub struct Phi3Vision {
    tokenizer: Tokenizer,
    vision_model: Session,
    text_embedding_model: Session,
    generation_model: Session,
    config: VisionConfig,
    
    last_analysis: Option<VisionAnalysisResult>,
    last_analysis_time: Instant,
    
    #[allow(dead_code)]
    analysis_sender: Option<mpsc::UnboundedSender<AnalysisRequest>>,
    #[allow(dead_code)]
    result_receiver: Option<mpsc::UnboundedReceiver<VisionAnalysisResult>>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct AnalysisRequest {
    image_data: Vec<u8>,
    width: u32,
    height: u32,
    prompt: String,
    timestamp: Instant,
}

impl Phi3Vision {
    pub async fn new(data_dir: &Path, config: VisionConfig) -> Result<Self> {
        tracing::info!("Initializing PHI-3.5 Vision models with CPU acceleration...");
        
        let _ = ort::init()
            .with_name("phi3_vision")
            .commit()
            .map_err(|e| {
                tracing::debug!("ONNX Runtime already initialized or failed: {:?}", e);
            });
        
        let tokenizer = Tokenizer::from_file(data_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Error loading tokenizer: {:?}", e))?;
        
        let create_session = |model_path: &str| -> Result<Session> {
            let builder = Session::builder()
                .map_err(|e| anyhow::anyhow!("Session builder error: {:?}", e))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("Optimization level error: {:?}", e))?
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .map_err(|e| anyhow::anyhow!("Execution provider error: {:?}", e))?;

            Ok(builder.commit_from_file(data_dir.join(model_path))
                .map_err(|e| anyhow::anyhow!("Model loading error: {:?}", e))?)
        };
        
        let vision_model = create_session("phi-3.5-v-instruct-vision.onnx")?;
        let text_embedding_model = create_session("phi-3.5-v-instruct-embedding.onnx")?;
        let generation_model = create_session("phi-3.5-v-instruct-text.onnx")?;
        
        tracing::info!("PHI-3.5 Vision models loaded successfully with CPU acceleration");
        
        Ok(Self {
            tokenizer,
            vision_model,
            text_embedding_model,
            generation_model,
            config,
            last_analysis: None,
            last_analysis_time: Instant::now(),
            analysis_sender: None,
            result_receiver: None,
        })
    }
    
    #[allow(dead_code)]
    pub fn start_analysis_task(&mut self) {
        let (analysis_sender, _analysis_receiver) = mpsc::unbounded_channel::<AnalysisRequest>();
        let (_result_sender, result_receiver) = mpsc::unbounded_channel::<VisionAnalysisResult>();
        
        self.analysis_sender = Some(analysis_sender);
        self.result_receiver = Some(result_receiver);
        
    }
    
    #[allow(dead_code)]
    pub fn should_analyze(&self) -> bool {
        self.last_analysis_time.elapsed() >= self.config.analysis_interval
    }
    
    pub fn analyze_frame_sync(
        &mut self,
        image_data: Vec<u8>,
        width: u32,
        height: u32,
        prompt: Option<String>,
    ) -> Result<VisionAnalysisResult> {
        let start_time = Instant::now();
        let prompt = prompt.unwrap_or_else(|| self.config.default_prompt.clone());
        
        tracing::debug!("Starting PHI-3.5 analysis for {}x{} image", width, height);
        
        let image = self.rgba_to_dynamic_image(image_data, width, height)?;
        
        let generated_text = self.generate_text_sync(&image, &prompt)?;
        
        let processing_time = start_time.elapsed();
        let result = VisionAnalysisResult {
            text: generated_text,
            timestamp: start_time,
            processing_time,
        };
        
        self.last_analysis = Some(result.clone());
        self.last_analysis_time = start_time;
        
        tracing::info!("PHI-3.5 analysis completed in {:?}: {}", processing_time, result.text);
        
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
        
        let image_buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;
        
        Ok(DynamicImage::ImageRgba8(image_buffer))
    }
    
    fn generate_text_sync(&mut self, image: &DynamicImage, text: &str) -> Result<String> {
        let (inputs_embeds, mut attention_mask) = {
            println!("🔍 Step 1: Getting image embedding...");
            let visual_features = self.get_image_embedding(image)?;
            
            println!("🔍 Step 2: Processing text prompt...");
            let prompt = self.format_chat_template(text);
            println!("🔍 Formatted prompt: {}", prompt);
            
            let encoding = self.tokenizer.encode(prompt, true)
                .map_err(|e| anyhow::anyhow!("Error encoding: {:?}", e))?;

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            println!("🔍 Token IDs length: {}", input_ids.len());
            println!("🔍 Token IDs: {:?}", &input_ids[..std::cmp::min(input_ids.len(), 10)]);
            
            let image_token_positions: Vec<usize> = input_ids.iter()
                .enumerate()
                .filter_map(|(i, &token)| if token == USER_TOKEN_ID { Some(i) } else { None })
                .collect();
            println!("🔍 Image token positions: {:?}", image_token_positions);
            
            let input_ids: Array2<i64> = Array2::from_shape_vec((1, input_ids.len()), input_ids)?;
            
            println!("🔍 Step 3: Getting text embedding...");
            let empty_image_features = Array2::zeros((0, 3072));
            let inputs_embeds: Array3<f32> = self.get_text_embedding(&input_ids, &empty_image_features)?;

            println!("🔍 Step 4: Creating attention mask...");
            let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&mask| mask as i64).collect();
            let attention_mask: Array2<i64> = Array2::from_shape_vec((1, attention_mask.len()), attention_mask)?;

            let image_token_position = input_ids.iter().position(|&id| id == USER_TOKEN_ID).unwrap_or(0);
            
            println!("🔍 Step 5: Merging embeddings...");
            let (combined_embeds, combined_mask) = self.merge_text_and_image_embeddings(
                &inputs_embeds, &attention_mask, &visual_features, image_token_position
            );
            
            println!("🔍 Step 6: Embeddings merged successfully, shape: {:?}", combined_embeds.shape());
            (combined_embeds, combined_mask)
        };

        let mut past_key_values: Vec<Array4<f32>> = vec![Array4::zeros((1, 32, 0, 96)); 64];
        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut next_inputs_embeds = inputs_embeds;
        
        for _ in 0..self.config.max_response_length {
            let model_inputs = {
                let mut model_inputs = ort::inputs![
                    "inputs_embeds" => TensorRef::from_array_view(&next_inputs_embeds)?,
                    "attention_mask" => TensorRef::from_array_view(&attention_mask)?,
                ];
                for i in 0..32 {
                    model_inputs.push((
                        format!("past_key_values.{}.key", i).into(),
                        TensorRef::from_array_view(&past_key_values[i * 2])?.into()
                    ));
                    model_inputs.push((
                        format!("past_key_values.{}.value", i).into(),
                        TensorRef::from_array_view(&past_key_values[i * 2 + 1])?.into()
                    ));
                }
                model_inputs
            };

            let model_outputs = self.generation_model.run(model_inputs)?;
            let logits = model_outputs["logits"].try_extract_array::<f32>()?.into_dimensionality::<ndarray::Ix3>()?;
            
            let next_token_id = logits
                .slice(s![0, -1, ..VOCAB_SIZE])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as i64;

            if next_token_id == EOS_TOKEN_ID {
                break;
            }

            generated_tokens.push(next_token_id);

            let mut new_past_key_values = Vec::with_capacity(64);
            for i in 0..32 {
                let key = model_outputs[format!("present.{}.key", i)]
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<ndarray::Ix4>()?
                    .to_owned();
                let value = model_outputs[format!("present.{}.value", i)]
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<ndarray::Ix4>()?
                    .to_owned();
                new_past_key_values.push(key);
                new_past_key_values.push(value);
            }
            
            drop(model_outputs);

            let new_token_id = Array2::from_elem((1, 1), next_token_id);
            let empty_image_features = Array2::zeros((0, 3072));
            next_inputs_embeds = self.get_text_embedding(&new_token_id, &empty_image_features)?;
            attention_mask = Array2::ones((1, attention_mask.shape()[1] + 1));
            past_key_values = new_past_key_values;
        }

        let output_ids: Vec<u32> = generated_tokens.iter().map(|&id| id as u32).collect();
        let generated_text = self.tokenizer.decode(&output_ids, false)
            .map_err(|e| anyhow::anyhow!("Decode error: {:?}", e))?;
        
        Ok(generated_text.trim().to_string())
    }
    
    fn get_image_embedding(&mut self, img: &DynamicImage) -> Result<Array3<f32>> {
        let image_processor = image_process::Phi3VImageProcessor::new();
        let result = image_processor.preprocess(img)?;
        
        println!("🔍 Vision model input shapes - pixel_values: {:?}, image_sizes: {:?}", 
                result.pixel_values.shape(), result.image_sizes.shape());
        
        let outputs = self.vision_model.run(ort::inputs![
            "pixel_values" => Tensor::from_array(result.pixel_values)?,
            "image_sizes" => Tensor::from_array(result.image_sizes)?,
        ])?;
        
        println!("🔍 Vision model ran successfully, extracting output...");
        let predictions_view = outputs["image_features"].try_extract_array::<f32>()?;
        println!("🔍 Raw output tensor shape: {:?}", predictions_view.shape());
        
        let image_features = match predictions_view.ndim() {
            2 => {
                println!("🔍 Converting 2D output to 3D");
                let shape = predictions_view.shape();
                let seq_len = shape[0];
                let hidden = shape[1];
                predictions_view.to_shape((1, seq_len, hidden))?.to_owned()
            },
            3 => {
                println!("🔍 Using 3D output as-is");
                predictions_view.into_dimensionality::<ndarray::Ix3>()?.to_owned()
            },
            4 => {
                println!("🔍 Converting 4D output to 3D");
                let dims = predictions_view.shape();
                let total_seq_len = dims[0] * dims[1] * dims[2];
                let hidden = dims[3];
                predictions_view.to_shape((1, total_seq_len, hidden))?.to_owned()
            },
            _ => {
                return Err(anyhow::anyhow!("Unexpected vision model output dimensionality: {}", predictions_view.ndim()));
            }
        };
        
        println!("🔍 Final image features shape: {:?}", image_features.shape());
        Ok(image_features)
    }
    
    fn get_text_embedding(&mut self, input_ids: &Array2<i64>, image_features: &Array2<f32>) -> Result<Array3<f32>> {
        println!("🔍 Text embedding input shapes - input_ids: {:?}, image_features: {:?}", 
                input_ids.shape(), image_features.shape());
        
        let outputs = self.text_embedding_model.run(ort::inputs![
            "input_ids" => TensorRef::from_array_view(input_ids)?,
            "image_features" => TensorRef::from_array_view(image_features)?,
        ])?;
        
        let inputs_embeds_view = outputs["inputs_embeds"].try_extract_array::<f32>()?;
        let text_embeds = inputs_embeds_view.into_dimensionality::<ndarray::Ix3>()?.to_owned();
        println!("🔍 Text embedding output shape - inputs_embeds: {:?}", text_embeds.shape());
        Ok(text_embeds)
    }
    
    fn merge_text_and_image_embeddings(
        &self,
        inputs_embeds: &Array3<f32>,
        attention_mask: &Array2<i64>,
        visual_features: &Array3<f32>,
        image_token_position: usize
    ) -> (Array3<f32>, Array2<i64>) {
        println!("🔍 Merging embeddings - text: {:?}, visual: {:?}, token_pos: {}", 
                inputs_embeds.shape(), visual_features.shape(), image_token_position);
        
        let mut combined_embeds = Array3::zeros((
            1,
            inputs_embeds.shape()[1] + visual_features.shape()[1],
            inputs_embeds.shape()[2]
        ));

        combined_embeds
            .slice_mut(s![.., ..image_token_position, ..])
            .assign(&inputs_embeds.slice(s![.., ..image_token_position, ..]));

        combined_embeds
            .slice_mut(s![.., image_token_position..(image_token_position + visual_features.shape()[1]), ..])
            .assign(visual_features);

        combined_embeds
            .slice_mut(s![.., (image_token_position + visual_features.shape()[1]).., ..])
            .assign(&inputs_embeds.slice(s![.., image_token_position.., ..]));

        let mut new_attention_mask = Array2::ones((1, attention_mask.shape()[1] + visual_features.shape()[1]));
        new_attention_mask
            .slice_mut(s![.., ..image_token_position])
            .assign(&attention_mask.slice(s![.., ..image_token_position]));
        new_attention_mask
            .slice_mut(s![.., (image_token_position + visual_features.shape()[1])..])
            .assign(&attention_mask.slice(s![.., image_token_position..]));

        (combined_embeds, new_attention_mask)
    }
    
    fn format_chat_template(&self, txt: &str) -> String {
        format!("<s><|user|>\n<|image_1|>\n{txt}<|end|>\n<|assistant|>\n")
    }
    
    #[allow(dead_code)]
    pub fn update_config(&mut self, config: VisionConfig) {
        self.config = config;
    }
}