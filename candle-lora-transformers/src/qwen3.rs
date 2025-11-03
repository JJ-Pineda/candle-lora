use candle_core::{DType, Device, Error, Module, Result, Tensor};
use candle_lora::{LoraConfig, LoraLinear, LoraLinearConfig, MultiLoraLinear};
use candle_nn::{kv_cache::KvCache, Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

#[derive(Debug, Clone)]
struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

impl Qwen3RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug)]
struct Qwen3MLP {
    gate_proj: MultiLoraLinear,
    up_proj: MultiLoraLinear,
    down_proj: MultiLoraLinear,
    act_fn: Activation,
    adapters: Vec<String>,
    active_adapter: Option<String>,
}

impl Qwen3MLP {
    fn new(cfg: &Config, vb: VarBuilder, lora_config: &LoraConfig) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        let gate_proj_base =
            candle_nn::linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let gate_proj_config = LoraLinearConfig::new(hidden_sz, intermediate_sz);
        let gate_proj_id = 0;
        let gate_proj_vb = vb.pp("gate_proj").pp("traced_lora_linear");
        let gate_proj = MultiLoraLinear::new(
            &gate_proj_base,
            &gate_proj_config,
            &lora_config,
            &gate_proj_vb,
            gate_proj_id,
        )?;

        let up_proj_base = candle_nn::linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let up_proj_config = LoraLinearConfig::new(hidden_sz, intermediate_sz);
        let up_proj_id = 0;
        let up_proj = MultiLoraLinear::new(
            &up_proj_base,
            &up_proj_config,
            &lora_config,
            &vb.pp("up_proj").pp("traced_lora_linear"),
            up_proj_id,
        )?;

        let down_proj_base =
            candle_nn::linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        let down_proj_config = LoraLinearConfig::new(intermediate_sz, hidden_sz);
        let down_proj_id = 0;
        let down_proj = MultiLoraLinear::new(
            &down_proj_base,
            &down_proj_config,
            &lora_config,
            &vb.pp("down_proj").pp("traced_lora_linear"),
            down_proj_id,
        )?;

        // Verify whether gate, up, or down projection actually found an adapter
        let (adapters, active_adapter) = if gate_proj.adapters.contains(&lora_config.name)
            || up_proj.adapters.contains(&lora_config.name)
            || down_proj.adapters.contains(&lora_config.name)
        {
            (
                vec![lora_config.name.clone()],
                Some(lora_config.name.clone()),
            )
        } else {
            (vec![], None)
        };

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            adapters,
            active_adapter,
        })
    }

    fn add_adapter(
        &mut self,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &LoraConfig,
    ) -> Result<()> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        let gate_proj_config = LoraLinearConfig::new(hidden_sz, intermediate_sz);
        let gate_proj_id = 0;
        let gate_proj_vb = vb.pp("gate_proj").pp("traced_lora_linear");
        self.gate_proj
            .add_adapter(&gate_proj_config, &lora_config, &gate_proj_vb, gate_proj_id)?;

        let up_proj_config = LoraLinearConfig::new(hidden_sz, intermediate_sz);
        let up_proj_id = 0;
        self.up_proj.add_adapter(
            &up_proj_config,
            &lora_config,
            &vb.pp("up_proj").pp("traced_lora_linear"),
            up_proj_id,
        )?;

        let down_proj_config = LoraLinearConfig::new(intermediate_sz, hidden_sz);
        let down_proj_id = 0;
        self.down_proj.add_adapter(
            &down_proj_config,
            &lora_config,
            &vb.pp("down_proj").pp("traced_lora_linear"),
            down_proj_id,
        )?;

        if self.gate_proj.adapters.contains(&lora_config.name)
            || self.up_proj.adapters.contains(&lora_config.name)
            || self.down_proj.adapters.contains(&lora_config.name)
        {
            self.adapters.push(lora_config.name.clone());
        }

        Ok(())
    }

    fn activate_adapter(&mut self, adapter_name: Option<&str>) {
        let _ = self.gate_proj.activate_adapter(adapter_name);
        let _ = self.up_proj.activate_adapter(adapter_name);
        let _ = self.down_proj.activate_adapter(adapter_name);

        if let Some(name) = adapter_name {
            if self.adapters.contains(&name.to_string()) {
                self.active_adapter = adapter_name.map(String::from);
            } else {
                self.active_adapter = None;
            }
        } else {
            self.active_adapter = None;
        }
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug)]
struct Qwen3Attention {
    // projections
    q_proj: MultiLoraLinear,
    k_proj: MultiLoraLinear,
    v_proj: MultiLoraLinear,
    o_proj: MultiLoraLinear,
    // norms
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    // hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    // utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: KvCache,
    adapters: Vec<String>,
    active_adapter: Option<String>,
}

impl Qwen3Attention {
    fn new(
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &LoraConfig,
    ) -> Result<Self> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * num_heads;

        let q_proj_base = candle_nn::linear_no_bias(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let q_proj_config = LoraLinearConfig::new(hidden_size, hidden_size);
        let q_proj_id = 0;
        let q_proj = MultiLoraLinear::new(
            &q_proj_base,
            &q_proj_config,
            &lora_config,
            &vb.pp("q_proj").pp("traced_lora_linear"),
            q_proj_id,
        )?;

        let k_proj_base =
            candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let k_proj_config = LoraLinearConfig::new(hidden_size, num_kv_heads * head_dim);
        let k_proj_id = 0;
        let k_proj = MultiLoraLinear::new(
            &k_proj_base,
            &k_proj_config,
            &lora_config,
            &vb.pp("k_proj").pp("traced_lora_linear"),
            k_proj_id,
        )?;

        let v_proj_base =
            candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let v_proj_config = LoraLinearConfig::new(hidden_size, num_kv_heads * head_dim);
        let v_proj_id = 0;
        let v_proj = MultiLoraLinear::new(
            &v_proj_base,
            &v_proj_config,
            &lora_config,
            &vb.pp("v_proj").pp("traced_lora_linear"),
            v_proj_id,
        )?;

        let o_proj_base = candle_nn::linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;
        let o_proj_config = LoraLinearConfig::new(hidden_size, hidden_size);
        let o_proj_id = 0;
        let o_proj = MultiLoraLinear::new(
            &o_proj_base,
            &o_proj_config,
            &lora_config,
            &vb.pp("o_proj").pp("traced_lora_linear"),
            o_proj_id,
        )?;

        let (adapters, active_adapter) = if q_proj.adapters.contains(&lora_config.name)
            || k_proj.adapters.contains(&lora_config.name)
            || v_proj.adapters.contains(&lora_config.name)
            || o_proj.adapters.contains(&lora_config.name)
        {
            (
                vec![lora_config.name.clone()],
                Some(lora_config.name.clone()),
            )
        } else {
            (vec![], None)
        };

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
        // The cache will grow in chunks of 512 tokens when needed.
        let kv_cache = KvCache::new(2, 512);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
            adapters,
            active_adapter,
        })
    }

    /// Repeats a key or value tensor for grouped query attention
    fn repeat_kv(&self, xs: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            Ok(xs)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
            Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
        }
    }

    fn add_adapter(
        &mut self,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &LoraConfig,
    ) -> Result<()> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * num_heads;

        let q_proj_config = LoraLinearConfig::new(hidden_size, hidden_size);
        let q_proj_id = 0;
        self.q_proj.add_adapter(
            &q_proj_config,
            &lora_config,
            &vb.pp("q_proj").pp("traced_lora_linear"),
            q_proj_id,
        )?;

        let k_proj_config = LoraLinearConfig::new(hidden_size, num_kv_heads * head_dim);
        let k_proj_id = 0;
        self.k_proj.add_adapter(
            &k_proj_config,
            &lora_config,
            &vb.pp("k_proj").pp("traced_lora_linear"),
            k_proj_id,
        )?;

        let v_proj_config = LoraLinearConfig::new(hidden_size, num_kv_heads * head_dim);
        let v_proj_id = 0;
        self.v_proj.add_adapter(
            &v_proj_config,
            &lora_config,
            &vb.pp("v_proj").pp("traced_lora_linear"),
            v_proj_id,
        )?;

        let o_proj_config = LoraLinearConfig::new(hidden_size, hidden_size);
        let o_proj_id = 0;
        self.o_proj.add_adapter(
            &o_proj_config,
            &lora_config,
            &vb.pp("o_proj").pp("traced_lora_linear"),
            o_proj_id,
        )?;

        if self.q_proj.adapters.contains(&lora_config.name)
            || self.k_proj.adapters.contains(&lora_config.name)
            || self.v_proj.adapters.contains(&lora_config.name)
            || self.o_proj.adapters.contains(&lora_config.name)
        {
            self.adapters.push(lora_config.name.clone());
        }

        Ok(())
    }

    fn activate_adapter(&mut self, adapter_name: Option<&str>) {
        let _ = self.q_proj.activate_adapter(adapter_name);
        let _ = self.k_proj.activate_adapter(adapter_name);
        let _ = self.v_proj.activate_adapter(adapter_name);
        let _ = self.o_proj.activate_adapter(adapter_name);

        if let Some(name) = adapter_name {
            if self.adapters.contains(&name.to_string()) {
                self.active_adapter = adapter_name.map(String::from);
            } else {
                self.active_adapter = None;
            }
        } else {
            self.active_adapter = None;
        }
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Proj
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per‑head RMSNorm
        let q_flat = q.flatten(0, 2)?; // (B*H, L, D) -> (BHL, D) after transpose later
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Accumulate KV cache
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // 6. GQA repeat_kv
        let k = self.repeat_kv(k, self.num_kv_groups)?;
        let v = self.repeat_kv(v, self.num_kv_groups)?;

        // 7. Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // 8. Output proj
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug)]
pub struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
    adapters: Vec<String>,
    active_adapter: Option<String>,
}

impl DecoderLayer {
    fn new(
        rotary: Arc<Qwen3RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &LoraConfig,
    ) -> Result<Self> {
        let self_attn = Qwen3Attention::new(rotary, cfg, vb.pp("self_attn"), lora_config)?;
        let mlp = Qwen3MLP::new(cfg, vb.pp("mlp"), lora_config)?;
        let ln1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let adapters = if self_attn.adapters.contains(&lora_config.name)
            || mlp.adapters.contains(&lora_config.name)
        {
            vec![lora_config.name.clone()]
        } else {
            vec![]
        };

        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
            adapters,
            active_adapter: Some(lora_config.name.clone()),
        })
    }

    fn add_adapter(
        &mut self,
        cfg: &Config,
        vb: &VarBuilder,
        lora_config: &LoraConfig,
    ) -> Result<()> {
        self.self_attn
            .add_adapter(cfg, vb.pp("self_attn"), lora_config)?;
        self.mlp.add_adapter(cfg, vb.pp("mlp"), lora_config)?;

        if self.self_attn.adapters.contains(&lora_config.name)
            || self.mlp.adapters.contains(&lora_config.name)
        {
            self.adapters.push(lora_config.name.clone());
            Ok(())
        } else {
            Err(Error::Msg("No LoRA weights detected!".to_string()))
        }
    }

    fn activate_adapter(&mut self, adapter_name: Option<&str>) -> Result<()> {
        if let Some(name) = adapter_name {
            if !self.adapters.contains(&name.to_string()) {
                return Err(Error::Msg(format!("Adapter '{}' does not exist!", name)));
            }
        }

        let _ = self.self_attn.activate_adapter(adapter_name);
        let _ = self.mlp.activate_adapter(adapter_name);
        self.active_adapter = adapter_name.map(String::from);

        Ok(())
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    pub layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
    adapters: Vec<String>,
    active_adapter: Option<String>,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder, lora_config: LoraConfig) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                rotary.clone(),
                cfg,
                vb_l.pp(i),
                &lora_config,
            )?);
        }

        let adapters = if layers[0].adapters.contains(&lora_config.name) {
            vec![lora_config.name.clone()]
        } else {
            vec![]
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            adapters,
            active_adapter: Some(lora_config.name.clone()),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    fn add_adapter(
        &mut self,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &LoraConfig,
    ) -> Result<()> {
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            self.layers[i].add_adapter(cfg, &vb_l.pp(i), lora_config)?;
        }

        if self.layers[0].adapters.contains(&lora_config.name) {
            self.adapters.push(lora_config.name.clone());
            Ok(())
        } else {
            Err(Error::Msg("No LoRA weights detected!".to_string()))
        }
    }

    fn activate_adapter(&mut self, adapter_name: Option<&str>) -> Result<()> {
        self.active_adapter = adapter_name.map(String::from);
        for i in 0..self.layers.len() {
            let _ = self.layers[i].activate_adapter(adapter_name)?;
        }

        Ok(())
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }
}

#[derive(Debug)]
pub struct ModelForCausalLM {
    pub base: Model,
    lm_head: LoraLinear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder, lora_config: LoraConfig) -> Result<Self> {
        let base = Model::new(cfg, vb.clone(), lora_config.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            let lm_head_base = candle_nn::Linear::new(base.embed_tokens.embeddings().clone(), None);
            let lm_head_config = LoraLinearConfig::new(cfg.hidden_size, cfg.vocab_size);
            let lm_head_id = 0;
            LoraLinear::new(
                &lm_head_base,
                &lm_head_config,
                &lora_config,
                &vb.pp("lm_head").pp("traced_lora_linear"),
                lm_head_id,
            )?
        } else {
            let lm_head_base =
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
            let lm_head_config = LoraLinearConfig::new(cfg.hidden_size, cfg.vocab_size);
            let lm_head_id = 0;
            LoraLinear::new(
                &lm_head_base,
                &lm_head_config,
                &lora_config,
                &vb.pp("lm_head").pp("traced_lora_linear"),
                lm_head_id,
            )?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}
