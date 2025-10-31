use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, Linear, VarBuilder};
use either::Either;

use crate::{
    frozenlinear::FrozenLinear, LinearLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
    Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraLinear {
    old: Arc<FrozenLinear>,
    ff_a: Linear,
    ff_b: Linear,
    scale: Option<f64>,
    dropout: Option<Arc<Dropout>>,
    merged: bool,
    prefix: String,
    id: usize,
    m: Option<Tensor>, // DoRA magnitude vector
}

#[derive(Clone, Debug)]
/// Configuration for LoraLinear
pub struct LoraLinearConfig {
    in_features: usize,
    out_features: usize,
}

impl LoraLinearConfig {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        LoraLinearConfig {
            in_features,
            out_features,
        }
    }
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        linear_config: &LoraLinearConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        let a = vb.pp(format!("a{id}")).get_with_hints(
            (config.rank, linear_config.in_features),
            "weight",
            init::DEFAULT_KAIMING_NORMAL,
        )?;
        let b = vb.pp(format!("b{id}")).get_with_hints(
            (linear_config.out_features, config.rank),
            "weight",
            init::ZERO,
        )?;

        // Try to load magnitude vector for DoRA (only if it exists in checkpoint)
        // VarBuilder.get() will create the tensor if it doesn't exist (with zeros)
        // Real DoRA magnitude vectors should have non-zero values, so we check for this
        let m: Option<Tensor> = if vb.contains_tensor(&format!("m{id}.weight")) {
            vb.pp(format!("m{id}"))
                .get(linear_config.out_features, "weight")
                .ok()
        } else {
            None
        };

        Ok(LoraLinear {
            old: Arc::new(FrozenLinear::new_from_linear(old)?),
            ff_a: Linear::new(a, None),
            ff_b: Linear::new(b, None),
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            dropout: config.dropout.map(|x| Arc::new(Dropout::new(x))),
            merged: false,
            prefix: vb.prefix(),
            id,
            m,
        })
    }
}

impl Merge for LoraLinear {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let ba = self
            .ff_b
            .weight()
            .matmul(self.ff_a.weight())
            .map_err(Either::Right)?;

        let scaled_ba = match self.scale {
            Some(scale) => ba.mul(scale).map_err(Either::Right)?,
            None => ba,
        };

        // For DoRA, apply magnitude normalization (row-wise)
        if let Some(ref m) = self.m {
            let w = self.old.weight();

            // Compute row-wise norms of W + scaled_BA
            // DoRA uses row-wise normalization per output channel (Section 3.1)
            let w_plus_ba = (w + &scaled_ba).map_err(Either::Right)?;
            let norms = w_plus_ba
                .sqr()
                .map_err(Either::Right)?
                .sum_keepdim(1) // Row-wise: sum over in_features dimension
                .map_err(Either::Right)?
                .sqrt()
                .map_err(Either::Right)?
                .squeeze(1)
                .map_err(Either::Right)?;

            // Normalized weight: m * (W + scaled_BA) / ||W + scaled_BA||_2
            let norm_scale = m.broadcast_div(&norms).map_err(Either::Right)?;
            let normalized = w_plus_ba
                .broadcast_mul(&norm_scale)
                .map_err(Either::Right)?;

            // Delta is normalized - W
            Ok((normalized - w).map_err(Either::Right)?)
        } else {
            // Standard LoRA
            Ok(scaled_ba)
        }
    }

    fn merge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if self.merged {
            Err(Either::Left(MergeError::AlreadyMerged))
        } else {
            self.old = Arc::new(
                FrozenLinear::new(
                    (self.old.weight() + self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias().cloned(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = true;
            Ok(())
        }
    }

    fn unmerge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if !self.merged {
            Err(Either::Left(MergeError::NotMerged))
        } else {
            self.old = Arc::new(
                FrozenLinear::new(
                    (self.old.weight() - self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias().cloned(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.merged {
            self.old.forward(input)
        } else {
            //No fan_in_fan_out so no weight.transpose(0,1)
            let mut result = self.old.forward(input)?;
            if let Some(scale) = self.scale {
                let input_new = if self.dropout.is_some() {
                    self.dropout.as_ref().unwrap().forward(input, true)?
                } else {
                    input.clone()
                };

                if let Some(ref m) = self.m {
                    // DoRA implementation (memory-efficient, row-wise normalization)
                    // DoRA: W' = m * (W + scale*BA) / ||W + scale*BA||_2
                    // where ||·||_2 is row-wise L2 norm (norm across in_features for each out_feature)
                    // Per DoRA paper Section 3.1: magnitude is computed per output channel (row)

                    // Note: `result` already contains W@x from line 169

                    // Compute adapter output: scale*B@A@x
                    let a_out = self.ff_a.forward(&input_new)?;
                    let ba_out = self.ff_b.forward(&a_out)?.mul(scale)?;

                    // Combined output: W@x + scale*B@A@x
                    let combined = (&result + &ba_out)?;

                    // Compute row-wise norms of W + scale*BA efficiently
                    // WITHOUT materializing the full BA matrix
                    // For each row i: ||W_i + scale*BA_i||² = sum_j (W_ij + scale*(BA)_ij)²
                    // We use: ||W_i + scale*BA_i||² = ||W_i||² + 2*scale*W_i·(BA)_i + scale²||BA_i||²

                    let w = self.old.weight(); // [out_features, in_features]
                    let b_weight = self.ff_b.weight(); // [out_features, rank]
                    let a_weight = self.ff_a.weight(); // [rank, in_features]

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
                    result = combined.broadcast_mul(&norm_scale)?;

                    // Add bias if present
                    if let Some(bias) = self.old.bias() {
                        result = result.broadcast_add(bias)?;
                    }
                } else {
                    // Standard LoRA implementation
                    let adapter_out = self
                        .ff_b
                        .forward(&self.ff_a.forward(&input_new)?)?
                        .mul(scale)?;
                    result = (result + adapter_out)?;
                }
            }
            Ok(result)
        }
    }
}

impl Saveable for LoraLinear {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.ff_a.weight().clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.ff_b.weight().clone(),
        );
        // Save magnitude vector if DoRA is used
        if let Some(ref m) = self.m {
            accum.insert(
                self.prefix.clone() + &format!(".m{}.weight", self.id),
                m.clone(),
            );
        }
    }
}

impl LinearLayerLike for LoraLinear {
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
    fn shape(&self) -> &Shape {
        self.old.shape()
    }
}
