use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_lora::LoraConfig;
use candle_lora_transformers::qwen3::{Config, Model, ModelForCausalLM};
use candle_lora_transformers::varbuilder_utils::from_mmaped_safetensors;
use candle_transformers::generation::LogitsProcessor;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;

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

fn collect_safetensors(dir: &str) -> Result<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort(); // ensure deterministic order
    anyhow::ensure!(!paths.is_empty(), "No safetensors found in {dir}");
    Ok(paths)
}

fn run_logits(model: &mut ModelForCausalLM, input_ids: &[u32], device: &Device) -> Result<Tensor> {
    let x = Tensor::new(input_ids, device)?.unsqueeze(0)?;
    let logits = model.forward(&x, 0)?.squeeze(0)?.squeeze(0)?;
    model.clear_kv_cache();
    Ok(logits)
}

fn token_to_string(tokenizer: &Tokenizer, id: u32) -> Result<Option<String>> {
    // Many tokenizers can decode one id at a time, but some BPEs produce
    // leading spaces/special joins. This is fine for a test; we only need
    // a minimal check that it resembles text.
    let s = tokenizer
        .decode(&[id], /*skip_special_tokens=*/ true)
        .map_err(anyhow::Error::msg)?;
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

#[test]
fn test_qwen3() -> Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let assets_dir = format!("{}/assets", manifest_dir);
    let base_dir = format!("{}/base_models/unsloth/Qwen3-8B", assets_dir);
    let cfg_path = format!("{}/config.json", base_dir);
    let tok_path = format!("{}/tokenizer.json", base_dir);
    let adapter_dir = format!("{}/pricebot/", assets_dir);

    // --- Load config & tokenizer ---
    let cfg_file = std::fs::File::open(&cfg_path)
        .with_context(|| format!("opening config json at {cfg_path}"))?;
    let cfg: Config = serde_json::from_reader(cfg_file)?;
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!(e))?;

    // --- Load base weights separately for comparison ---
    let base_weight_files = collect_safetensors(&base_dir)?;
    let (device, dtype) = get_device_and_dtype();
    let vb = from_mmaped_safetensors(&base_weight_files, dtype, &device, false)?;

    // --- Build model ---
    let default_lora_cfg = LoraConfig::new(8, 16.0, None);
    let model = Arc::new(RwLock::new(Model::new(&cfg, vb.clone(), default_lora_cfg)?));
    let mut model = ModelForCausalLM::from_base(Arc::clone(&model), &cfg, vb)?;

    // --- Tiny generation loop (greedy / top-p+temp) ---
    let prompt =
        "<|im_start|>user\n\nHello, who are you?\n\n<|im_start|>assistant\n\n<think>\n\n</think>";
    let mut ids = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let logits_base = run_logits(&mut model, &ids, &device)?.to_dtype(DType::F32)?;

    // Now load adapter
    let adapter_files = collect_safetensors(&adapter_dir)?;
    let vb = from_mmaped_safetensors(&adapter_files, dtype, &device, false)?;

    let real_lora_cfg = LoraConfig::new_with_name(256, 512.0, None, "dora0");
    let _ = model.add_adapter(&cfg, vb, &real_lora_cfg)?;
    let _ = model.activate_adapter(Some("dora0"))?;

    let logits_lora = run_logits(&mut model, &ids, &device)?.to_dtype(DType::F32)?;
    let diff = (&logits_lora - &logits_base)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;

    assert_ne!(diff, 0f32);

    // --- Tiny generation loop (greedy / top-p+temp) ---
    // Small, deterministic settings for testability.
    let seed: u64 = 42;
    let temperature: Option<f64> = Some(0.7);
    let top_p: Option<f64> = Some(0.9);
    let mut lp = LogitsProcessor::new(seed, temperature, top_p);

    let eos_id = tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .copied()
        .unwrap_or_else(|| {
            println!("Could not find EOS token!");
            u32::MAX
        });

    let max_new_tokens = 10usize; // keep short for test runtime
    let mut generated_text = String::new();

    println!("Starting generation loop...");
    for step in 0..max_new_tokens {
        // Only feed the full context once; then one token at a time using KV-cache.
        let ctx_len = if step == 0 { ids.len() } else { 1 };
        let start_pos = ids.len().saturating_sub(ctx_len);
        let ctx = &ids[start_pos..];

        let input = Tensor::new(ctx, &device)?.unsqueeze(0)?; // [1, ctx_len]
        let logits = model
            .forward(&input, start_pos)
            .context("forward pass")?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;

        let next_id = lp.sample(&logits).context("sampling")?;
        ids.push(next_id);
        if next_id == eos_id {
            break;
        }

        // Best-effort incremental decode
        if let Some(tok) = token_to_string(&tokenizer, next_id)? {
            generated_text.push_str(&tok);
        }
    }

    // Clear KV cache
    model.clear_kv_cache();

    // --- Basic assertions ---
    // Ensure we *actually* produced something beyond the prompt.
    anyhow::ensure!(
        generated_text.trim().len() >= 1,
        "Model produced empty output text."
    );

    println!("{}", generated_text);

    Ok(())
}
