// New llama-cpp-2 based modules
pub mod llama_multimodal;
pub mod llama_image_process;
pub mod llama_download;

// Export new types
pub use llama_multimodal::{LlamaMultimodal, LlamaMultimodalConfig, LlamaMultimodalAnalysisResult};