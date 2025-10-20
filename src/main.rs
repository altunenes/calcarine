use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ShaderHotReload};
use cuneus::compute::{ BindGroupLayoutType, create_bind_group_layout, create_external_texture_bind_group};
use std::path::PathBuf;
use std::time::Instant;
use cuneus::prelude::*;
use log::{info, warn, error, debug};

mod fastvlm;

// Analysis request type for future llama-cpp-2 integration
#[allow(dead_code)]
#[derive(Debug)]
enum AnalysisRequest {
    SingleFrame {
        image_data: Vec<u8>,
        width: u32,
        height: u32,
        prompt: String,
    },
    VideoChunk {
        frames: Vec<(Vec<u8>, u32, u32)>,
        prompt: String,
    },
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CalcarineParams {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue_shift: f32,
    gamma: f32,
    vignette: f32,
    noise: f32,
    _padding: u32,
}

impl UniformProvider for CalcarineParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
struct Calcarine {
    base: RenderKit,
    params_uniform: UniformBinding<CalcarineParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    compute_pipeline: wgpu::ComputePipeline,
    output_texture: cuneus::TextureManager,
    capture_output_texture: Option<cuneus::TextureManager>,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    frame_count: u32,
    hot_reload: cuneus::ShaderHotReload,
    
    last_analysis_time: Instant,
    llm_enabled: bool,
    llm_prompt: String,
    analysis_interval_seconds: f32,
    llm_resolution_scale: f32,
    llm_processing: bool,
    
    analysis_sender: Option<std::sync::mpsc::Sender<AnalysisRequest>>,
    result_receiver: Option<std::sync::mpsc::Receiver<fastvlm::LlamaMultimodalAnalysisResult>>,
    last_analysis_result: Option<fastvlm::LlamaMultimodalAnalysisResult>,
    
    video_mode_enabled: bool,
    video_chunk_interval: f32, // seconds between video chunks
    last_video_chunk_time: Instant,
    video_frame_buffer: Vec<(Vec<u8>, u32, u32)>, // Buffer for video frames
    
    // Model downloading state
    downloading_models: bool,
    
    // Frame capture retry tracking
    capture_retry_count: u32,
    capture_retry_start: Option<Instant>,
    
}

impl Calcarine {
    fn recreate_compute_resources(&mut self, core: &Core) {
        let input_texture_view;
        let input_sampler;
        
        if self.base.using_video_texture {
            if let Some(ref video_manager) = self.base.video_texture_manager {
                let texture_manager = video_manager.texture_manager();
                input_texture_view = &texture_manager.view;
                input_sampler = &texture_manager.sampler;
            } else if let Some(ref texture_manager) = self.base.texture_manager {
                input_texture_view = &texture_manager.view;
                input_sampler = &texture_manager.sampler;
            } else {
                panic!("No texture available for compute shader input");
            }
        } else if self.base.using_webcam_texture {
            if let Some(ref webcam_manager) = self.base.webcam_texture_manager {
                let texture_manager = webcam_manager.texture_manager();
                input_texture_view = &texture_manager.view;
                input_sampler = &texture_manager.sampler;
            } else if let Some(ref texture_manager) = self.base.texture_manager {
                input_texture_view = &texture_manager.view;
                input_sampler = &texture_manager.sampler;
            } else {
                panic!("No texture available for compute shader input");
            }
        } else if let Some(ref texture_manager) = self.base.texture_manager {
            input_texture_view = &texture_manager.view;
            input_sampler = &texture_manager.sampler;
        } else {
            panic!("No texture available for compute shader input");
        }
        
        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            wgpu::TextureFormat::Rgba16Float,
            &self.base.texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Color Projection Output",
        );
        
        
        let view_output = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.compute_bind_group = create_external_texture_bind_group(
            &core.device,
            &self.compute_bind_group_layout,
            input_texture_view,
            input_sampler,
            &view_output,
            "Color Projection Compute",
        );
    }
    
    fn capture_frame(&mut self, core: &Core, time: f32) -> Result<Vec<u8>, wgpu::SurfaceError> {
        let width = ((core.size.width as f32 * self.llm_resolution_scale) as u32).max(320);
        let height = ((core.size.height as f32 * self.llm_resolution_scale) as u32).max(240);
        let (capture_texture, output_buffer) = self.base.create_capture_texture(
            &core.device,
            width,
            height
        );
        
        let align = 256;
        let unpadded_bytes_per_row = width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Capture Encoder"),
        });
        
        if self.capture_output_texture.is_none() {
            self.capture_output_texture = Some(cuneus::compute::create_output_texture(
                &core.device,
                width,
                height,
                wgpu::TextureFormat::Rgba16Float,
                &self.base.texture_bind_group_layout,
                wgpu::AddressMode::ClampToEdge,
                wgpu::FilterMode::Linear,
                "Capture Output",
            ));
        }
        
        let capture_output = self.capture_output_texture.as_ref().unwrap();
        
        let capture_compute_bind_group = if self.base.using_video_texture {
            if let Some(ref video_manager) = self.base.video_texture_manager {
                let texture_manager = video_manager.texture_manager();
                cuneus::compute::create_external_texture_bind_group(
                    &core.device,
                    &self.compute_bind_group_layout,
                    &texture_manager.view,
                    &texture_manager.sampler,
                    &capture_output.view,
                    "Capture Compute",
                )
            } else {
                return Err(wgpu::SurfaceError::Lost);
            }
        } else if self.base.using_webcam_texture {
            if let Some(ref webcam_manager) = self.base.webcam_texture_manager {
                let texture_manager = webcam_manager.texture_manager();
                cuneus::compute::create_external_texture_bind_group(
                    &core.device,
                    &self.compute_bind_group_layout,
                    &texture_manager.view,
                    &texture_manager.sampler,
                    &capture_output.view,
                    "Capture Compute",
                )
            } else {
                return Err(wgpu::SurfaceError::Lost);
            }
        } else if let Some(ref texture_manager) = self.base.texture_manager {
            cuneus::compute::create_external_texture_bind_group(
                &core.device,
                &self.compute_bind_group_layout,
                &texture_manager.view,
                &texture_manager.sampler,
                &capture_output.view,
                "Capture Compute",
            )
        } else {
            return Err(wgpu::SurfaceError::Lost);
        };
        
        let capture_time_uniform = UniformBinding::new(
            &core.device,
            "Capture Time Uniform",
            cuneus::compute::ComputeTimeUniform {
                time,
                delta: 1.0/60.0,
                frame: self.frame_count,
                _padding: 0,
            },
            &self.time_bind_group_layout,
            0,
        );
        capture_time_uniform.update(&core.queue);
        
        let compute_width = width.div_ceil(16);
        let compute_height = height.div_ceil(16);
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Capture Process Texture Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &capture_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &capture_compute_bind_group, &[]);
            
            compute_pass.dispatch_workgroups(compute_width, compute_height, 1);
        }
        
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &capture_output.bind_group, &[]);
            
            render_pass.draw(0..4, 0..1);
        }
        
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        
        core.queue.submit(Some(encoder.finish()));
        
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            // Ignore send errors - the receiver might be dropped if we timed out
            debug!("GPU buffer mapping callback triggered");
            match tx.send(result) {
                Ok(_) => {
                    debug!("GPU buffer result sent successfully");
                },
                Err(_) => {
                    debug!("GPU buffer receiver was dropped (timeout expected)");
                }
            }
        });
        
        // Use a short blocking wait to give the GPU operation time to complete
        debug!("Polling GPU device with short wait...");
        core.device.poll(wgpu::Maintain::Wait);
        
        // Try to receive with a very short timeout
        debug!("Waiting briefly for GPU buffer results...");
        match rx.recv_timeout(std::time::Duration::from_millis(1)) {
            Ok(result) => {
                debug!("GPU buffer operation completed successfully!");
                result.unwrap();
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                debug!("GPU buffer operation timed out");
                return Err(wgpu::SurfaceError::Timeout);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                warn!("GPU buffer channel disconnected");
                return Err(wgpu::SurfaceError::Lost);
            }
        }
        
        let padded_data = buffer_slice.get_mapped_range().to_vec();
        let mut unpadded_data = Vec::with_capacity((width * height * 4) as usize);
        
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        
        Ok(unpadded_data)
    }
    
    fn capture_frame_sync(&mut self, core: &Core, time: f32) -> Result<Vec<u8>, wgpu::SurfaceError> {
        // Synchronous version for fallback
        info!("Using synchronous frame capture as fallback");
        let width = ((core.size.width as f32 * self.llm_resolution_scale) as u32).max(320);
        let height = ((core.size.height as f32 * self.llm_resolution_scale) as u32).max(240);
        let (capture_texture, output_buffer) = self.base.create_capture_texture(
            &core.device,
            width,
            height
        );
        
        let align = 256;
        let unpadded_bytes_per_row = width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sync Capture Encoder"),
        });
        
        if self.capture_output_texture.is_none() {
            self.capture_output_texture = Some(cuneus::compute::create_output_texture(
                &core.device,
                width,
                height,
                wgpu::TextureFormat::Rgba16Float,
                &self.base.texture_bind_group_layout,
                wgpu::AddressMode::ClampToEdge,
                wgpu::FilterMode::Linear,
                "Sync Capture Output",
            ));
        }
        
        let capture_output = self.capture_output_texture.as_ref().unwrap();
        
        let capture_compute_bind_group = if self.base.using_video_texture {
            if let Some(ref video_manager) = self.base.video_texture_manager {
                let texture_manager = video_manager.texture_manager();
                cuneus::compute::create_external_texture_bind_group(
                    &core.device,
                    &self.compute_bind_group_layout,
                    &texture_manager.view,
                    &texture_manager.sampler,
                    &capture_output.view,
                    "Sync Capture Compute",
                )
            } else {
                return Err(wgpu::SurfaceError::Lost);
            }
        } else if self.base.using_webcam_texture {
            if let Some(ref webcam_manager) = self.base.webcam_texture_manager {
                let texture_manager = webcam_manager.texture_manager();
                cuneus::compute::create_external_texture_bind_group(
                    &core.device,
                    &self.compute_bind_group_layout,
                    &texture_manager.view,
                    &texture_manager.sampler,
                    &capture_output.view,
                    "Sync Capture Compute",
                )
            } else {
                return Err(wgpu::SurfaceError::Lost);
            }
        } else if let Some(ref texture_manager) = self.base.texture_manager {
            cuneus::compute::create_external_texture_bind_group(
                &core.device,
                &self.compute_bind_group_layout,
                &texture_manager.view,
                &texture_manager.sampler,
                &capture_output.view,
                "Sync Capture Compute",
            )
        } else {
            return Err(wgpu::SurfaceError::Lost);
        };
        
        let capture_time_uniform = UniformBinding::new(
            &core.device,
            "Sync Capture Time Uniform",
            cuneus::compute::ComputeTimeUniform {
                time,
                delta: 1.0/60.0,
                frame: self.frame_count,
                _padding: 0,
            },
            &self.time_bind_group_layout,
            0,
        );
        capture_time_uniform.update(&core.queue);
        
        let compute_width = width.div_ceil(16);
        let compute_height = height.div_ceil(16);
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sync Capture Process Texture Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &capture_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &capture_compute_bind_group, &[]);
            
            compute_pass.dispatch_workgroups(compute_width, compute_height, 1);
        }
        
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Sync Capture Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &capture_output.bind_group, &[]);
            
            render_pass.draw(0..4, 0..1);
        }
        
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        
        core.queue.submit(Some(encoder.finish()));
        
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        
        // Use blocking wait for synchronous version
        core.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let padded_data = buffer_slice.get_mapped_range().to_vec();
        let mut unpadded_data = Vec::with_capacity((width * height * 4) as usize);
        
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        
        Ok(unpadded_data)
    }
    
    fn handle_export(&mut self, core: &Core) {
        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            if let Ok(data) = self.capture_frame_for_export(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = cuneus::save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            self.base.export_manager.complete_export();
        }
    }
    
    fn capture_frame_for_export(&mut self, core: &Core, time: f32) -> Result<Vec<u8>, wgpu::SurfaceError> {
        let settings = self.base.export_manager.settings();
        let (capture_texture, output_buffer) = self.base.create_capture_texture(
            &core.device,
            settings.width,
            settings.height
        );
        
        let align = 256;
        let unpadded_bytes_per_row = settings.width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Export Capture Encoder"),
        });
        
        self.base.time_uniform.data.time = time;
        self.base.time_uniform.update(&core.queue);
        
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Export Capture Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.output_texture.bind_group, &[]);
            
            render_pass.draw(0..4, 0..1);
        }
        
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(settings.height),
                },
            },
            wgpu::Extent3d {
                width: settings.width,
                height: settings.height,
                depth_or_array_layers: 1,
            },
        );
        
        core.queue.submit(Some(encoder.finish()));
        
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        core.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let padded_data = buffer_slice.get_mapped_range().to_vec();
        let mut unpadded_data = Vec::with_capacity((settings.width * settings.height * 4) as usize);
        
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        
        Ok(unpadded_data)
    }
    
    fn init_llama_multimodal() -> Option<fastvlm::LlamaMultimodal> {
        info!("Setting up Llama-cpp-2 multimodal...");
        
        match fastvlm::llama_download::setup_model_config() {
            Ok((model_path, mmproj_path)) => {
                info!("Found models:");
                info!("  Model: {}", model_path);
                info!("  Projection: {}", mmproj_path);
                
                let config = fastvlm::LlamaMultimodalConfig {
                    model_path,
                    mmproj_path,
                    max_response_length: 50,
                    default_prompt: "Describe this image in less than 10 words".to_string(),
                    n_ctx: 4096,
                    n_threads: 4,
                    video_chunk_size: 10,
                    chunk_overlap_frames: 2,
                };
                
                // Initialize the multimodal system
                let rt = tokio::runtime::Runtime::new().ok()?;
                match rt.block_on(fastvlm::LlamaMultimodal::new(config)) {
                    Ok(llama) => {
                        info!("Llama-cpp-2 multimodal initialized successfully!");
                        Some(llama)
                    },
                    Err(e) => {
                        error!("Failed to initialize multimodal system: {}", e);
                        None
                    }
                }
            },
            Err(e) => {
                warn!("Model setup failed: {}", e);
                None
            }
        }
    }
    
    fn handle_llm_analysis(&mut self, core: &Core) {
        if !self.llm_enabled || self.analysis_sender.is_none() {
            return;
        }
        
        if let Some(receiver) = &self.result_receiver {
            if let Ok(result) = receiver.try_recv() {
                info!("LLM Analysis Result: {}", result.text);
                info!("Processing time: {:.2}s", result.processing_time.as_secs_f32());
                self.last_analysis_result = Some(result);
                self.llm_processing = false;
            }
        }
        
        if self.llm_processing {
            return;
        }
        
        let should_analyze = self.last_analysis_time.elapsed().as_secs_f32() >= self.analysis_interval_seconds;
        
        if should_analyze {
            info!("Starting AI analysis - pausing intensive GPU graphics for optimal AI performance...");
            self.last_analysis_time = Instant::now();
            self.llm_processing = true;
            
            
            let current_time = self.base.controls.get_time(&self.base.start_time);
            
            match self.capture_frame(core, current_time) {
                Ok(frame_data) => {
                    // Reset retry tracking on successful capture
                    self.capture_retry_start = None;
                    self.capture_retry_count = 0;
                    
                    let captured_width = ((core.size.width as f32 * self.llm_resolution_scale) as u32).max(320);
                    let captured_height = ((core.size.height as f32 * self.llm_resolution_scale) as u32).max(240);
                    debug!("Frame captured successfully: {}x{} pixels", 
                           captured_width, captured_height);
                    
                    let prompt = if self.llm_prompt.trim().is_empty() {
                        "Describe this image in 10 words or less.".to_string()
                    } else {
                        self.llm_prompt.clone()
                    };
                    
                    info!("Processing your image - results will appear shortly...");
                    
                    if let Some(_sender) = &self.analysis_sender {
                        let request = if self.video_mode_enabled {
                            // Add frame to video buffer
                            self.video_frame_buffer.push((frame_data, captured_width, captured_height));
                            
                            // Check if it's time to process video chunk
                            let should_process_chunk = self.last_video_chunk_time.elapsed().as_secs_f32() >= self.video_chunk_interval;
                            
                            if should_process_chunk && !self.video_frame_buffer.is_empty() {
                                self.last_video_chunk_time = Instant::now();
                                let chunk_frames = std::mem::take(&mut self.video_frame_buffer);
                                let video_prompt = format!("Analyze this video sequence of {} frames: {}", chunk_frames.len(), prompt);
                                Some(AnalysisRequest::VideoChunk { 
                                    frames: chunk_frames,
                                    prompt: video_prompt,
                                })
                            } else {
                                None // Don't send request yet, accumulating frames
                            }
                        } else {
                            // Single frame analysis
                            Some(AnalysisRequest::SingleFrame {
                                image_data: frame_data,
                                width: captured_width,
                                height: captured_height,
                                prompt,
                            })
                        };
                        
                        if let Some(request) = request {
                            // Send the request to background thread
                            if let Some(ref sender) = self.analysis_sender {
                                match sender.send(request) {
                                    Ok(_) => {
                                        debug!("Request sent to background MTMD thread");
                                    }
                                    Err(e) => {
                                        error!("Failed to send request to background thread: {}", e);
                                        self.llm_processing = false;
                                    }
                                }
                            } else {
                                warn!("No background thread available for processing");
                                self.llm_processing = false;
                            }
                        } else {
                            // Video mode: frame added to buffer, not processing yet
                            self.llm_processing = false;
                        }
                    }
                },
                Err(wgpu::SurfaceError::Timeout) => {
                    // Track retry attempts and use fallback if needed
                    if self.capture_retry_start.is_none() {
                        self.capture_retry_start = Some(Instant::now());
                        self.capture_retry_count = 0;
                    }
                    
                    self.capture_retry_count += 1;
                    let elapsed = self.capture_retry_start.unwrap().elapsed();
                    
                    const MAX_RETRIES: u32 = 3;
                    const FALLBACK_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(100);
                    
                    if self.capture_retry_count >= MAX_RETRIES || elapsed >= FALLBACK_TIMEOUT {
                        warn!("Async capture failed after {} retries in {:.2}s, falling back to sync capture", 
                                self.capture_retry_count, elapsed.as_secs_f32());
                        
                        // Reset retry tracking
                        self.capture_retry_start = None;
                        self.capture_retry_count = 0;
                        
                        // Try synchronous capture as fallback
                        match self.capture_frame_sync(core, current_time) {
                            Ok(frame_data) => {
                                let captured_width = ((core.size.width as f32 * self.llm_resolution_scale) as u32).max(320);
                                let captured_height = ((core.size.height as f32 * self.llm_resolution_scale) as u32).max(240);
                                debug!("Sync frame captured successfully: {}x{} pixels", 
                                       captured_width, captured_height);
                                
                                let prompt = if self.llm_prompt.trim().is_empty() {
                                    "Describe this image in 10 words or less.".to_string()
                                } else {
                                    self.llm_prompt.clone()
                                };
                                
                                // Send request to background thread
                                if let Some(ref sender) = self.analysis_sender {
                                    let request = AnalysisRequest::SingleFrame { 
                                        image_data: frame_data,
                                        width: captured_width,
                                        height: captured_height,
                                        prompt,
                                    };
                                    
                                    match sender.send(request) {
                                        Ok(_) => {
                                            debug!("Sync request sent to background MTMD thread");
                                        }
                                        Err(e) => {
                                            error!("Failed to send sync request to background thread: {}", e);
                                            self.llm_processing = false;
                                        }
                                    }
                                } else {
                                    warn!("No background thread available for processing");
                                    self.llm_processing = false;
                                }
                            }
                            Err(e) => {
                                error!("Sync capture also failed: {:?}", e);
                                self.llm_processing = false;
                            }
                        }
                    } else {
                        // Continue retrying
                        debug!("Frame capture in progress, retry {}/{} ({:.2}s)...", 
                                self.capture_retry_count, MAX_RETRIES, elapsed.as_secs_f32());
                        // Don't set llm_processing = false, keep it active to retry
                    }
                },
                Err(e) => {
                    error!("Failed to capture frame for analysis: {:?}", e);
                    self.llm_processing = false;
                }
            }
        }
    }
}

impl ShaderManager for Calcarine {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });
        
        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform, 
            "Color Projection"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "Color Projection Params"
        );
        
        
        let compute_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });
        
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Calcarine Params",
            CalcarineParams {
                brightness: 0.0,
                contrast: 1.0,
                saturation: 1.0,
                hue_shift: 0.0,
                gamma: 1.0,
                vignette: 0.0,
                noise: 0.0,
                _padding: 0,
            },
            &params_bind_group_layout,
            0,
        );
        
        let compute_time_uniform = UniformBinding::new(
            &core.device,
            "Compute Time Uniform",
            cuneus::compute::ComputeTimeUniform {
                time: 0.0,
                delta: 0.0,
                frame: 0,
                _padding: 0,
            },
            &time_bind_group_layout,
            0,
        );
        
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Color Projection Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shader/calcarine.wgsl").into()),
        });
        
        let hot_reload = ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shader/calcarine.wgsl"),
            cs_module.clone(),
            "process_texture",
        ).expect("Failed to initialize hot reload");
        
        let base = RenderKit::new(
            core,
            include_str!("../shader/vertex.wgsl"),
            include_str!("../shader/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.config.width,
            core.config.height,
            wgpu::TextureFormat::Rgba16Float,
            &texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Color Projection Output",
        );
        
        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &params_bind_group_layout,
                &compute_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Calcarine Texture Processing Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("process_texture"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base.texture_manager.as_ref().unwrap().view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&base.texture_manager.as_ref().unwrap().sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_texture.view),
                },
            ],
            label: Some("Compute Bind Group"),
        });
        
        // Initialize Llama-cpp-2 multimodal (may be None if models not available)
        info!("Initializing Llama-cpp-2 multimodal...");
        let llama_multimodal = Self::init_llama_multimodal();
        let llm_enabled = llama_multimodal.is_some();
        
        // Set up background thread for multimodal processing  
        let (analysis_sender, result_receiver) = if let Some(llama_instance) = llama_multimodal {
            let mut llama_instance = llama_instance;
            info!("Setting up analysis pipeline with background thread...");
            let (request_tx, request_rx) = std::sync::mpsc::channel::<AnalysisRequest>();
            let (result_tx, result_rx) = std::sync::mpsc::channel::<fastvlm::LlamaMultimodalAnalysisResult>();
            
            // Spawn background thread for MTMD processing
            std::thread::spawn(move || {
                info!("Background MTMD processing thread started");
                while let Ok(request) = request_rx.recv() {
                    debug!("Background thread received request: {:?}", 
                             match &request {
                                 AnalysisRequest::SingleFrame { width, height, .. } => 
                                     format!("SingleFrame({}x{})", width, height),
                                 AnalysisRequest::VideoChunk { frames, .. } => 
                                     format!("VideoChunk({} frames)", frames.len()),
                             });
                    match request {
                        AnalysisRequest::SingleFrame { image_data, width, height, prompt } => {
                            info!("Processing frame with REAL llama-cpp-2 multimodal in background...");
                            match llama_instance.analyze_frame_sync(image_data, width, height, Some(prompt)) {
                                Ok(result) => {
                                    info!("REAL Multimodal analysis completed: {}", result.text);
                                    let _ = result_tx.send(result);
                                }
                                Err(e) => {
                                    error!("REAL Multimodal analysis failed: {}", e);
                                    let error_result = fastvlm::LlamaMultimodalAnalysisResult {
                                        text: format!("Analysis error: {}", e),
                                        timestamp: std::time::Instant::now(),
                                        processing_time: std::time::Duration::from_millis(100),
                                    };
                                    let _ = result_tx.send(error_result);
                                }
                            }
                        }
                        AnalysisRequest::VideoChunk { frames, prompt } => {
                            info!("Processing video chunk with {} frames using REAL multimodal in background...", frames.len());
                            
                            // Add all frames to the video buffer in the multimodal instance
                            for (frame_data, _width, _height) in &frames {
                                llama_instance.add_video_frame(frame_data.clone());
                            }
                            
                            // For now, analyze the first frame with video context
                            if let Some((first_frame, width, height)) = frames.first() {
                                let video_prompt = format!("This is a video sequence with {} frames. {}. Describe the movement, action, or changes you see.", frames.len(), prompt);
                                match llama_instance.analyze_frame_sync(first_frame.clone(), *width, *height, Some(video_prompt)) {
                                    Ok(result) => {
                                        info!("REAL Video chunk analysis completed: {}", result.text);
                                        // Get video status for debugging
                                        let (buffer_size, chunk_count) = llama_instance.get_video_status();
                                        debug!("Video buffer: {} frames, {} chunks processed", buffer_size, chunk_count);
                                        let _ = result_tx.send(result);
                                    }
                                    Err(e) => {
                                        error!("REAL Video chunk analysis failed: {}", e);
                                        let error_result = fastvlm::LlamaMultimodalAnalysisResult {
                                            text: format!("Video chunk analysis error: {}", e),
                                            timestamp: std::time::Instant::now(),
                                            processing_time: std::time::Duration::from_millis(100),
                                        };
                                        let _ = result_tx.send(error_result);
                                    }
                                }
                            }
                        }
                    }
                }
                info!("Background MTMD processing thread terminated");
            });
            
            (Some(request_tx), Some(result_rx))
        } else {
            (None, None)
        };
        
        if llm_enabled {
            info!("Llama-cpp-2 multimodal initialized successfully!");
        } else {
            warn!("Multimodal system initialization failed - AI features disabled");
        }
        
        let mut result = Self {
            base,
            params_uniform,
            compute_time_uniform,
            compute_pipeline,
            output_texture,
            compute_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            compute_bind_group,
            frame_count: 0,
            hot_reload,
            last_analysis_time: Instant::now(),
            llm_enabled,
            llm_prompt: "Describe this image in 10 words or less.".to_string(),
            analysis_interval_seconds: 12.0,
            llm_resolution_scale: 0.3,
            llm_processing: false,
            analysis_sender,
            result_receiver,
            last_analysis_result: None,
            capture_output_texture: None,
            
            // New video analysis fields
            video_mode_enabled: false,
            video_chunk_interval: 5.0, // 5 seconds between video chunks
            last_video_chunk_time: Instant::now(),
            video_frame_buffer: Vec::new(),
            
            // Model downloading
            downloading_models: false,
            
            // Frame capture retry tracking
            capture_retry_count: 0,
            capture_retry_start: None,
        };
        
        result.recreate_compute_resources(core);
        
        result
    }
    
    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            info!("Reloading compute shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
                let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.compute_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            self.compute_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Calcarine Texture Processing Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("process_texture"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        }
        let video_updated = if self.base.using_video_texture {
            self.base.update_video_texture(core, &core.queue)
        } else {
            false
        };
        let webcam_updated = if self.base.using_webcam_texture {
            self.base.update_webcam_texture(core, &core.queue)
        } else {
            false
        };
        if video_updated || webcam_updated {
            self.recreate_compute_resources(core);
        }
        
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        
        self.handle_llm_analysis(core);
        
        self.base.fps_tracker.update();
    }

    
    fn resize(&mut self, core: &Core) {
        debug!("Resizing to {:?}", core.size);
        self.recreate_compute_resources(core);
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let should_start_export = false;
        let export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        
        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();
        
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                
                egui::Window::new("Calcarine Settings")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                    ShaderControls::render_media_panel(
                        ui,
                        &mut controls_request,
                        using_video_texture,
                        video_info,
                        using_hdri_texture,
                        hdri_info,
                        using_webcam_texture,
                        webcam_info
                    );
                        ui.separator();
                        
                        egui::CollapsingHeader::new("Image Adjustments")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.brightness, -1.0..=1.0).text("Brightness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.contrast, 0.0..=2.0).text("Contrast")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.saturation, 0.0..=2.0).text("Saturation")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=3.0).text("Gamma")).changed();

                                if ui.button("Reset Basic").clicked() {
                                    params.brightness = 0.0;
                                    params.contrast = 1.0;
                                    params.saturation = 1.0;
                                    params.gamma = 1.0;
                                    changed = true;
                                }
                            });
                        
                        egui::CollapsingHeader::new("Effects")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.hue_shift, 0.0..=6.28).text("Hue Shift")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.vignette, 0.0..=1.0).text("Vignette")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.noise, 0.0..=1.0).text("Noise")).changed();
                                
                                if ui.button("Reset Effects").clicked() {
                                    params.hue_shift = 0.0;
                                    params.vignette = 0.0;
                                    params.noise = 0.0;
                                    changed = true;
                                }
                            });
                        
                        ui.separator();
                        
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        
                        egui::CollapsingHeader::new("🤖 Multimodal AI Analysis (Llama-cpp-2)")
                            .default_open(false)
                            .show(ui, |ui| {
                                if self.analysis_sender.is_some() {
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut self.llm_enabled, "Enable AI Analysis");
                                        let next_analysis_in = self.analysis_interval_seconds - self.last_analysis_time.elapsed().as_secs_f32();
                                        if self.llm_processing {
                                            ui.label("🧠 Processing...");
                                        } else if self.llm_enabled && next_analysis_in > 0.0 {
                                            ui.label(format!("Next in {:.1}s", next_analysis_in));
                                        } else if self.llm_enabled {
                                            ui.label("🔄 Ready");
                                        }
                                    });
                                    
                                    // NEW: Analysis Mode Selection
                                    ui.horizontal(|ui| {
                                        ui.label("Mode:");
                                        ui.radio_value(&mut self.video_mode_enabled, false, "📷 Single Frame");
                                        ui.radio_value(&mut self.video_mode_enabled, true, "🎬 Video Chunks");
                                    });
                                    
                                    if self.video_mode_enabled {
                                        ui.add(egui::Slider::new(&mut self.video_chunk_interval, 2.0..=15.0)
                                            .text("Video Chunk Interval (seconds)"));
                                        
                                        let (buffered_frames, _chunk_count) = (self.video_frame_buffer.len(), 0); // We'd need to track chunk count
                                        ui.label(format!("📹 Buffered frames: {}", buffered_frames));
                                        
                                        if ui.button("🗑️ Clear Video Buffer").clicked() {
                                            self.video_frame_buffer.clear();
                                            self.last_video_chunk_time = Instant::now();
                                        }
                                    } else {
                                        ui.add(egui::Slider::new(&mut self.analysis_interval_seconds, 3.0..=30.0)
                                            .text("Analysis Interval (seconds)"));
                                    }
                                    
                                    ui.add(egui::Slider::new(&mut self.llm_resolution_scale, 0.25..=1.0)
                                        .text("Resolution Scale (lower = faster)")
                                        .show_value(true));
                                    
                                    ui.horizontal(|ui| {
                                        ui.label("Custom Prompt:");
                                        ui.text_edit_singleline(&mut self.llm_prompt);
                                    });
                                    
                                    ui.add_space(5.0);
                                    if ui.button("🔍 Analyze Now").clicked() && self.llm_enabled && !self.llm_processing {
                                        info!("Manual analysis triggered");
                                        self.last_analysis_time = Instant::now() - std::time::Duration::from_secs(self.analysis_interval_seconds as u64);
                                    }
                                } else {
                                    ui.heading("⚠️ Multimodal AI Unavailable");
                                    ui.label("Llama-cpp-2 models not found or failed to load");
                                    ui.label("Required: .gguf model file + mmproj-*.gguf");
                                    ui.add_space(5.0);
                                    
                                    let model_dir = fastvlm::llama_download::get_default_model_dir();
                                    ui.label(format!("📂 Expected location: {}", model_dir.display()));
                                    ui.add_space(5.0);
                                    
                                    if self.downloading_models {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label("📥 Downloading models... (check console for progress)");
                                        });
                                    } else {
                                        if ui.button("📥 Download Models Automatically").clicked() {
                                            info!("Starting automatic model download...");
                                            self.downloading_models = true;
                                            
                                            // Start download in background thread
                                            std::thread::spawn(|| {
                                                let rt = tokio::runtime::Runtime::new().unwrap();
                                                rt.block_on(async {
                                                    if let Err(e) = fastvlm::llama_download::download_models().await {
                                                        error!("Download failed: {}", e);
                                                    } else {
                                                        info!("Download completed! Please restart the application.");
                                                    }
                                                });
                                            });
                                        }
                                        
                                        ui.label("📖 Or check console for manual download instructions");
                                        ui.label("🔄 Restart the app after downloading models");
                                    }
                                }
                            });
                        
                        if self.analysis_sender.is_some() {
                            egui::CollapsingHeader::new("💬 AI Analysis Results")
                                .default_open(self.last_analysis_result.is_some())
                                .show(ui, |ui| {
                                    if let Some(analysis) = &self.last_analysis_result {
                                        let elapsed = analysis.timestamp.elapsed();
                                        ui.label(format!("⏱️ {:.1}s ago • 🚀 Processed in {:.2}s", 
                                                       elapsed.as_secs_f32(), 
                                                       analysis.processing_time.as_secs_f32()));
                                        
                                        ui.add_space(5.0);
                                        egui::Frame::new()
                                            .fill(egui::Color32::from_rgba_premultiplied(40, 40, 50, 180))
                                            .corner_radius(egui::CornerRadius::same(6))
                                            .inner_margin(egui::Margin::same(10))
                                            .show(ui, |ui| {
                                                ui.add(egui::Label::new(
                                                    egui::RichText::new(&analysis.text)
                                                        .size(13.0)
                                                        .color(egui::Color32::WHITE)
                                                ).wrap());
                                            });
                                    } else if self.llm_enabled {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label("⏳ Initializing AI analysis - this may take a moment...");
                                        });
                                        ui.label("Your analysis request is queued and will process shortly");
                                    } else {
                                        ui.label("Analysis is disabled");
                                    }
                                });
                        }
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            self.recreate_compute_resources(core);
        }
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);
        if controls_request.load_media_path.is_some() || controls_request.start_webcam {
            self.recreate_compute_resources(core);
        }
        if self.base.handle_hdri_requests(core, &controls_request) {
            self.recreate_compute_resources(core);
        }
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        self.compute_time_uniform.data.time = current_time;
        self.compute_time_uniform.data.delta = 1.0/60.0;
        self.compute_time_uniform.data.frame = self.frame_count;
        self.compute_time_uniform.update(&core.queue);
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        let width = core.size.width.div_ceil(16);
        let height = core.size.height.div_ceil(16);
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Process Texture Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.output_texture.bind_group, &[]);
            
            render_pass.draw(0..4, 0..1);
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        self.frame_count = self.frame_count.wrapping_add(1);
        
        Ok(())
    }
    
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        
        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                eprintln!("Failed to load dropped file: {:?}", e);
            } else {
                self.recreate_compute_resources(core);
            }
            return true;
        }
        
        false
    }
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // On Windows, allocate a console for GUI apps to show download progress
    #[cfg(target_os = "windows")]
    {
        use std::ffi::CString;
        unsafe extern "system" {
            fn AllocConsole() -> i32;
            fn SetConsoleTitleA(title: *const i8) -> i32;
        }
        unsafe {
            AllocConsole();
            let title = CString::new("Calcarine - Setup").unwrap();
            SetConsoleTitleA(title.as_ptr());
        }
    }
    
    cuneus::gst::init()?;
    
    env_logger::init();
    
    info!("Starting Calcarine with FastVLM integration");
    
    // On Windows, hide the console after downloads are complete
    #[cfg(target_os = "windows")]
    {
        unsafe extern "system" {
            fn FreeConsole() -> i32;
        }
        unsafe {
            FreeConsole();
        }
    }
    
    let (app, event_loop) = cuneus::ShaderApp::new("Calcarine", 800, 600);
    app.run(event_loop, |core| {
        Calcarine::init(core)
    })
}