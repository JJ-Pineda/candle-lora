use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_lora::LoraConfig;
use candle_lora_transformers::qwen3::{Config, Model, ModelForCausalLM, ModelTokenizer};
use candle_lora_transformers::varbuilder_utils::from_mmaped_safetensors;
use std::sync::{Arc, RwLock};

const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// Gets the appropriate device and dtype based on hardware availability.
/// Priority: CUDA (BF16) > Metal (F16) > CPU (F32)
fn get_device_and_dtype() -> (Device, DType) {
    if let Ok(device) = Device::new_cuda(0) {
        (device, DType::BF16)
    } else {
        if let Ok(device) = Device::new_metal(0) {
            (device, DType::F16)
        } else {
            (Device::Cpu, DType::F32)
        }
    }
}

/// Collects all safetensors files from a directory in sorted order
fn collect_safetensors(dir: &str) -> Result<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    anyhow::ensure!(!paths.is_empty(), "No safetensors found in {dir}");
    Ok(paths)
}

/// Runs a forward pass through the model with the given input_ids and clears the KV cache
fn run_logits(model: &mut ModelForCausalLM, input_ids: &[u32], device: &Device) -> Result<Tensor> {
    let x = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    let logits = model.forward(&x, 0, None)?.squeeze(0)?.squeeze(0)?;
    model.clear_kv_cache();
    Ok(logits)
}

/// Builds a prompt in the Qwen3 chat format
fn build_prompt(msg: &str) -> String {
    format!("<|im_start|>user\n\n{msg}\n\n<|im_start|>assistant\n\n<think>\n\n</think>")
}

/// Loads the base model config and VarBuilder from disk
fn get_base_vb(base_dir: &str, device: &Device, dtype: DType) -> Result<(Config, VarBuilder)> {
    let cfg_path = format!("{}/config.json", base_dir);
    let cfg_file = std::fs::File::open(&cfg_path)
        .with_context(|| format!("opening config json at {cfg_path}"))?;
    let cfg: Config = serde_json::from_reader(cfg_file)?;

    let base_weight_files = collect_safetensors(base_dir)?;
    let vb = from_mmaped_safetensors(&base_weight_files, dtype, device, false)?;

    Ok((cfg, vb))
}

/// Tokenizes test prompts with left padding
fn get_test_ids(msgs: &[&str], tokenizer: &ModelTokenizer) -> Result<Vec<Vec<u32>>> {
    let prompts: Vec<String> = msgs.iter().map(|m| build_prompt(m)).collect();
    let prompt_refs: Vec<&str> = prompts.iter().map(|s| s.as_str()).collect();

    let ids = tokenizer
        .encode(prompt_refs, Some("left"))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(ids)
}

/// Tests DoRA adapter loading and activation using ModelForCausalLM::from_base
fn test_qwen3_dora(
    tokenizer: &ModelTokenizer,
    device: &Device,
    dtype: DType,
    cfg: &Config,
    model: Arc<RwLock<Model>>,
    vb: VarBuilder,
    adapter_dir: &str,
) -> Result<()> {
    println!("\n=== Testing DoRA adapter (using ModelForCausalLM::from_base) ===");
    println!("Using device: {:?}, dtype: {:?}", device, dtype);

    // Create ModelForCausalLM using from_base (this tests from_base method)
    let mut model = ModelForCausalLM::from_base(model, cfg, vb)?;

    // Get test input
    let ids_vec = get_test_ids(&["Hello, who are you?", "What is 2+2?"], tokenizer)?;
    let ids: Vec<&[u32]> = ids_vec.iter().map(|v| v.as_slice()).collect();

    // Run forward pass with base model (no adapter)
    println!("Running inference with base model...");
    let logits_base = run_logits(&mut model, ids[0], device)?.to_dtype(DType::F32)?;

    // Load DoRA adapter
    println!("Loading DoRA adapter...");
    let adapter_files = collect_safetensors(adapter_dir)?;
    let adapter_vb = from_mmaped_safetensors(&adapter_files, dtype, device, false)?;
    let dora_cfg = LoraConfig::new_with_name(256, 512.0, None, "dora0");
    model.add_adapter(cfg, adapter_vb, &dora_cfg)?;

    // Activate adapter and run forward pass
    println!("Activating adapter and running inference...");
    model.activate_adapter(Some("dora0"))?;
    let logits_lora = run_logits(&mut model, ids[0], device)?.to_dtype(DType::F32)?;

    // Verify adapter changes the output
    let diff = (&logits_lora - &logits_base)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;

    anyhow::ensure!(diff > 0.0, "DoRA adapter did not change the output!");

    println!("✓ DoRA adapter test passed\n");
    Ok(())
}

/// Tests text generation using ModelForCausalLM::new
fn test_qwen3_generation(tokenizer: &ModelTokenizer, mut model: ModelForCausalLM) -> Result<()> {
    println!("\n=== Testing text generation ===");

    // Get test inputs
    let ids_vec = get_test_ids(&["Hello, who are you?", "What is 2+2?"], tokenizer)?;
    let ids: Vec<&[u32]> = ids_vec.iter().map(|v| v.as_slice()).collect();

    // Generate responses
    println!("Generating responses (max 10 tokens)...");
    let output_ids = model.generate(&ids, Some(0.7), Some(0.9), Some(10), None)?;

    anyhow::ensure!(
        output_ids.len() == 2,
        "Expected 2 generated sequences, got {}",
        output_ids.len()
    );

    // Decode and verify outputs
    let generated_texts = tokenizer
        .decode(output_ids, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    for (i, text) in generated_texts.iter().enumerate() {
        anyhow::ensure!(
            !text.trim().is_empty(),
            "Model produced empty output text for sequence {}",
            i
        );
        println!("Sequence {}: {}", i, text);
    }

    println!("✓ Generation test passed\n");
    Ok(())
}

/// Composite test that runs both DoRA and generation tests
#[test]
fn test_qwen3() -> Result<()> {
    // Setup common paths
    let assets_dir = format!("{MANIFEST_DIR}/assets");
    let base_dir = format!("{assets_dir}/base_models/unsloth/Qwen3-8B");
    let tok_path = format!("{base_dir}/tokenizer.json");
    let adapter_dir = format!("{MANIFEST_DIR}/assets/pricebot/");

    // Initialize shared resources (only once to ensure consistency)
    let tokenizer = ModelTokenizer::new(&tok_path, 151654)?;
    let (device, dtype) = get_device_and_dtype();
    let (cfg, vb) = get_base_vb(&base_dir, &device, dtype)?;

    // Test 1: DoRA adapter with from_base
    // Create ModelForCausalLM using new
    let temp_lora_cfg = LoraConfig::new(8, 16.0, None);
    let model = ModelForCausalLM::new(&cfg, vb.clone(), temp_lora_cfg)?;
    test_qwen3_dora(
        &tokenizer,
        &device,
        dtype,
        &cfg,
        Arc::clone(&model.base),
        vb.clone(),
        &adapter_dir,
    )?;

    // Test 2: Text generation with new
    test_qwen3_generation(&tokenizer, model)?;

    Ok(())
}
