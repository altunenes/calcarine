//! Test the official MTMD CLI example with our models

use std::ffi::CString;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/Users/enes/Library/Caches/Calcarine/models/gemma-3-4b-it-Q4_K_M.gguf";
    let mmproj_path = "/Users/enes/Library/Caches/Calcarine/models/mmproj-F16.gguf";
    let image_path = "/Users/enes/Desktop/download.png";
    let prompt = "What is in the picture?";
    let n_predict = 20;
    let n_threads = 4;
    let n_tokens = NonZeroU32::new(4096).unwrap();

    // Validate files exist
    if !Path::new(model_path).exists() {
        eprintln!("Error: Model file not found: {}", model_path);
        return Err("Model file not found".into());
    }
    if !Path::new(mmproj_path).exists() {
        eprintln!("Error: Multimodal projection file not found: {}", mmproj_path);
        return Err("Multimodal projection file not found".into());
    }
    if !Path::new(image_path).exists() {
        eprintln!("Error: Image file not found: {}", image_path);
        return Err("Image file not found".into());
    }

    println!("Loading model: {}", model_path);

    // Initialize backend (exactly like official mtmd.rs)
    let backend = LlamaBackend::init()?;

    // Setup model parameters (exactly like official mtmd.rs)
    let mut model_params = LlamaModelParams::default();
    model_params = model_params.with_n_gpu_layers(1_000_000); // Use all layers on GPU

    // Load model (exactly like official mtmd.rs)
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    // Create context (exactly like official mtmd.rs)
    let context_params = LlamaContextParams::default()
        .with_n_threads(n_threads)
        .with_n_batch(1)
        .with_n_ctx(Some(n_tokens));
    let mut context = model.new_context(&backend, context_params)?;

    // Create sampler (exactly like official mtmd.rs)
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    println!("Model loaded successfully");
    println!("Loading mtmd projection: {}", mmproj_path);

    // Initialize MTMD context (exactly like official mtmd.rs)
    let mtmd_params = MtmdContextParams {
        use_gpu: true,
        print_timings: true,
        n_threads,
        media_marker: CString::new("<start_of_image>")?,
    };

    let mtmd_ctx = MtmdContext::init_from_file(mmproj_path, &model, &mtmd_params)?;

    // Get chat template (exactly like official mtmd.rs)
    let chat_template = model.chat_template(None)?;

    // Create batch (exactly like official mtmd.rs)
    let mut batch = LlamaBatch::new(n_tokens.get() as usize, 1);

    // Add media marker if not present (exactly like official mtmd.rs)
    let mut full_prompt = prompt.to_string();
    let media_marker = "<start_of_image>";
    if !full_prompt.contains(media_marker) {
        full_prompt.push_str(media_marker);
    }

    // Load image bitmap (exactly like official mtmd.rs)
    println!("Loading image: {}", image_path);
    let bitmap = MtmdBitmap::from_file(&mtmd_ctx, image_path)?;
    let bitmaps = vec![bitmap];

    // Create user message (exactly like official mtmd.rs)
    let msg = LlamaChatMessage::new("user".to_string(), full_prompt)?;
    let chat = vec![msg.clone()];

    println!("Evaluating message: {:?}", msg);

    // Format the message using chat template (exactly like official mtmd.rs)
    let formatted_prompt = model.apply_chat_template(&chat_template, &chat, true)?;

    let input_text = MtmdInputText {
        text: formatted_prompt,
        add_special: true,
        parse_special: true,
    };

    let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

    println!("Tokenizing with {} bitmaps", bitmap_refs.len());

    // Tokenize the input (exactly like official mtmd.rs)
    let chunks = mtmd_ctx.tokenize(input_text, &bitmap_refs)?;
    println!("Tokenization complete, {} chunks created", chunks.len());

    // Evaluate chunks (exactly like official mtmd.rs - line 189)
    println!("🔄 Starting chunk evaluation...");
    let n_past = chunks.eval_chunks(&mtmd_ctx, &mut context, 0, 0, 1, true)?;
    println!("✅ Chunk evaluation completed!");

    // Generate response (exactly like official mtmd.rs)
    let mut generated_tokens = Vec::new();
    let max_predict = if n_predict < 0 { i32::MAX } else { n_predict };

    println!("🔄 Starting token generation...");

    for i in 0..max_predict {
        // Sample next token
        let token = sampler.sample(&context, 0);
        generated_tokens.push(token);
        sampler.accept(token);

        // Check for end of generation
        if model.is_eog_token(token) {
            println!();
            break;
        }

        // Print token
        let piece = model.token_to_str(token, Special::Tokenize)?;
        print!("{piece}");
        std::io::stdout().flush()?;

        // Prepare next batch
        batch.clear();
        batch.add(token, n_past + i + 1, &[0], true)?;

        // Decode
        context.decode(&mut batch)?;
    }

    println!("\n✅ Generation completed!");

    Ok(())
}