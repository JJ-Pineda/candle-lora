use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Error, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::{frozenlinear::FrozenLinear, LinearLayerLike, LoraConfig, LoraLinearConfig};

#[derive(Debug, Clone)]
pub struct MultiLoraLinear {
    old: Arc<FrozenLinear>,
    ff_a: HashMap<String, Linear>,
    ff_b: HashMap<String, Linear>,
    scale: HashMap<String, Option<f64>>,
    m: HashMap<String, Option<Tensor>>, // DoRA magnitude vector
    pub adapters: Vec<String>,
    active_adapter: Option<String>,
}

impl MultiLoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        linear_config: &LoraLinearConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        let (a, b, m) = MultiLoraLinear::get_adapter_weights(linear_config, config, vb, id);

        let (a_map, b_map, m_map, scale_map, active_adapter) = if a.is_some() && b.is_some() {
            let a_map = HashMap::from([(config.name.clone(), Linear::new(a.unwrap(), None))]);
            let b_map = HashMap::from([(config.name.clone(), Linear::new(b.unwrap(), None))]);
            let m_map = HashMap::from([(config.name.clone(), m)]);

            let scale_map = if config.rank > 0 {
                let scale = Some(config.alpha / config.rank as f64);
                HashMap::from([(config.name.clone(), scale)])
            } else {
                HashMap::from([(config.name.clone(), None)])
            };

            (a_map, b_map, m_map, scale_map, Some(config.name.clone()))
        } else {
            (
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                None,
            )
        };

        let adapters = if !a_map.is_empty() {
            vec![config.name.clone()]
        } else {
            vec![]
        };

        Ok(MultiLoraLinear {
            old: Arc::new(FrozenLinear::new_from_linear(old)?),
            ff_a: a_map,
            ff_b: b_map,
            scale: scale_map,
            m: m_map,
            adapters,
            active_adapter,
        })
    }

    pub fn add_adapter(
        &mut self,
        linear_config: &LoraLinearConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> Result<()> {
        // Verify that adapter name doesn't already exist to avoid overwriting
        if self.ff_a.contains_key(&config.name) {
            return Err(Error::Msg(format!(
                "Adapter '{}' already exists!",
                &config.name
            )));
        }

        // Rank must be 1 or more
        if config.rank < 1 {
            return Err(Error::Msg(
                "Adapter rank must be greater than or equal to 1!".to_string(),
            ));
        }

        let (a, b, m) = MultiLoraLinear::get_adapter_weights(linear_config, config, vb, id);

        if a.is_some() && b.is_some() {
            let scale = Some(config.alpha / config.rank as f64);

            self.ff_a
                .insert(config.name.clone(), Linear::new(a.unwrap(), None));
            self.ff_b
                .insert(config.name.clone(), Linear::new(b.unwrap(), None));
            self.m.insert(config.name.clone(), m);
            self.scale.insert(config.name.clone(), scale);
            self.adapters.push(config.name.clone());
        }

        Ok(())
    }

    fn get_adapter_weights(
        linear_config: &LoraLinearConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> (Option<Tensor>, Option<Tensor>, Option<Tensor>) {
        let a: Option<Tensor> = if vb.contains_tensor(&format!("a{id}.weight")) {
            vb.pp(format!("a{id}"))
                .get((config.rank, linear_config.in_features), "weight")
                .ok()
        } else {
            None
        };

        let b: Option<Tensor> = if vb.contains_tensor(&format!("b{id}.weight")) {
            vb.pp(format!("b{id}"))
                .get((linear_config.out_features, config.rank), "weight")
                .ok()
        } else {
            None
        };

        let m: Option<Tensor> = if vb.contains_tensor(&format!("m{id}.weight")) {
            vb.pp(format!("m{id}"))
                .get(linear_config.out_features, "weight")
                .ok()
        } else {
            None
        };

        (a, b, m)
    }

    pub fn activate_adapter(&mut self, adapter_name: Option<&str>) -> Result<()> {
        match adapter_name {
            Some(name) => {
                if !self.adapters.contains(&name.to_string()) {
                    self.active_adapter = None;
                    Ok(())
                } else {
                    self.active_adapter = Some(name.to_string());
                    Ok(())
                }
            }
            None => {
                self.active_adapter = None;
                Ok(())
            }
        }
    }
}

impl Module for MultiLoraLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.adapters.is_empty() || self.active_adapter.is_none() {
            self.old.forward(input)
        } else {
            // We ensure that the "active_adapter" always exists otherwise it's None
            // I.e. "unwrap" is safe here
            let adapter_name = self.active_adapter.clone().unwrap();

            // Store bias to add at the end
            let bias = self.old.bias();

            let scale = *self.scale.get(&adapter_name).unwrap();

            if let Some(scale) = scale {
                let input_new = input.clone();

                let a = self.ff_a.get(&adapter_name).unwrap();
                let b = self.ff_b.get(&adapter_name).unwrap();
                let m = self.m.get(&adapter_name).unwrap();

                if let Some(m) = m {
                    // DoRA implementation (memory-efficient, row-wise normalization)
                    // DoRA: W' = m * (W + scale*BA) / ||W + scale*BA||_2
                    // where ||·||_2 is row-wise L2 norm (norm across in_features for each out_feature)
                    // Per DoRA paper Section 3.1: magnitude is computed per output channel (row)

                    // IMPORTANT: Compute W@x WITHOUT bias (bias added at the end)
                    let w = self.old.weight(); // [out_features, in_features]
                    let w_linear = Linear::new(w.clone(), None);
                    let w_out = w_linear.forward(&input_new)?; // W@x without bias

                    // Compute adapter output: scale*B@A@x
                    let a_out = a.forward(&input_new)?;
                    let ba_out = b.forward(&a_out)?.mul(scale)?;

                    // Combined output: W@x + scale*B@A@x (still no bias)
                    let combined = (&w_out + &ba_out)?;

                    // Compute row-wise norms of W + scale*BA efficiently
                    // WITHOUT materializing the full BA matrix
                    // For each row i: ||W_i + scale*BA_i||² = sum_j (W_ij + scale*(BA)_ij)²
                    // We use: ||W_i + scale*BA_i||² = ||W_i||² + 2*scale*W_i·(BA)_i + scale²||BA_i||²

                    let b_weight = b.weight(); // [out_features, rank]
                    let a_weight = a.weight(); // [rank, in_features]

                    // Compute ||W_i||² for each row (sum over in_features dimension)
                    let w_norm_sq = w.sqr()?.sum_keepdim(1)?; // [out_features, 1]

                    // Compute ||BA_i||² efficiently using the kernel trick
                    // For each row i: ||BA_i||² = (BA)(BA)^T_ii = (B @ A @ A^T @ B^T)_ii
                    // We compute: B @ (A @ A^T) @ B^T and take diagonal elements
                    let aat = a_weight.matmul(&a_weight.t()?)?; // [rank, rank]
                    let b_aat = b_weight.matmul(&aat)?; // [out_features, rank]
                    let ba_norm_sq = (b_aat * b_weight)?.sum_keepdim(1)?; // [out_features, 1]

                    // Compute 2*W_i·(BA)_i efficiently
                    // For each row i: W_i · (BA)_i = sum_j W_ij * (BA)_ij
                    // where (BA)_ij = sum_k B_ik * A_kj
                    // So: sum_j W_ij * sum_k B_ik * A_kj = sum_k B_ik * sum_j W_ij * A_kj
                    //                                    = sum_k B_ik * (W @ A^T)_ik
                    //                                    = (W @ A^T) ⊙ B summed over k for each row
                    let wa_t = w.matmul(&a_weight.t()?)?; // [out_features, rank]
                    let cross = (wa_t * b_weight)?.sum_keepdim(1)?.mul(2.0 * scale)?; // [out_features, 1]

                    // ||W + scale*BA||² = ||W||² + 2*scale*W·BA + scale²||BA||²
                    let norms = (w_norm_sq + cross + ba_norm_sq.mul(scale * scale)?)?
                        .sqrt()?
                        .squeeze(1)?; // [out_features]

                    // Apply DoRA row-wise: m * combined / norms
                    // m is [out_features], norms is [out_features]
                    // This normalizes each output feature (row) independently
                    let norm_scale = m.broadcast_div(&norms)?; // [out_features]
                    let mut result = combined.broadcast_mul(&norm_scale)?;

                    // Add bias at the very end
                    if let Some(b) = bias {
                        result = result.broadcast_add(b)?;
                    }
                    Ok(result)
                } else {
                    // Standard LoRA implementation
                    // Compute W@x without bias
                    let w = self.old.weight();
                    let w_linear = Linear::new(w.clone(), None);
                    let mut result = w_linear.forward(&input_new)?;

                    // Add adapter output
                    let adapter_out = b.forward(&a.forward(&input_new)?)?.mul(scale)?;
                    result = (result + adapter_out)?;

                    // Add bias at the end
                    if let Some(b) = bias {
                        result = result.broadcast_add(b)?;
                    }
                    Ok(result)
                }
            } else {
                self.old.forward(input)
            }
        }
    }
}
