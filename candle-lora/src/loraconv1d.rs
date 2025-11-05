use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Conv1d, Conv1dConfig, Dropout, VarBuilder};
use either::Either;

use crate::{
    frozenconv::FrozenConv1d, Conv1dLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
    Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraConv1d {
    old: Arc<FrozenConv1d>,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    dropout: Option<Arc<Dropout>>,
    merged: bool,
    prefix: String,
    id: usize,
    m: Option<Tensor>,
}

#[derive(Clone, Debug)]
/// Configuration for LoraConv1d. Other configurations are inherited from the `Conv1d` struct.
pub struct LoraConv1dConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl LoraConv1dConfig {
    pub fn new(kernel_size: usize, in_channels: usize, out_channels: usize) -> Self {
        LoraConv1dConfig {
            in_channels,
            out_channels,
            kernel_size,
        }
    }
}

impl LoraConv1d {
    pub fn new(
        old: &dyn Conv1dLayerLike,
        conv_config: &LoraConv1dConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        let a = vb.pp(format!("a{id}")).get_with_hints(
            (
                config.rank * conv_config.kernel_size,
                conv_config.in_channels * conv_config.kernel_size,
            ),
            "weight",
            init::DEFAULT_KAIMING_NORMAL,
        )?;
        let b = vb.pp(format!("b{id}")).get_with_hints(
            (
                conv_config.out_channels / old.config().groups * conv_config.kernel_size,
                config.rank * conv_config.kernel_size,
            ),
            "weight",
            init::ZERO,
        )?;

        // Try to load magnitude vector for DoRA
        let m: Option<Tensor> = if vb.contains_tensor(&format!("m{id}.weight")) {
            vb.pp(format!("m{id}"))
                .get(
                    (
                        conv_config.out_channels / old.config().groups,
                        conv_config.in_channels * conv_config.kernel_size,
                    ),
                    "weight",
                )
                .ok()
        } else {
            None
        };

        Ok(LoraConv1d {
            old: Arc::new(FrozenConv1d::new_from_conv1d(old)?),
            a,
            b,
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

impl Merge for LoraConv1d {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let ba = self
            .b
            .matmul(&self.a)
            .map_err(Either::Right)?
            .reshape(self.old.weight().shape())
            .map_err(Either::Right)?;

        let scaled_ba = match self.scale {
            Some(scale) => ba.mul(scale).map_err(Either::Right)?,
            None => ba,
        };

        // For DoRA, apply magnitude normalization (row-wise)
        // Conv1d weights are [out_channels, in_channels, kernel_size]
        // We flatten to [out_channels, in_channels * kernel_size] for row-wise normalization
        if let Some(ref m) = self.m {
            let w = self.old.weight();
            let original_shape = w.shape().clone();

            // Flatten weight to 2D: [out_channels, in_channels * kernel_size]
            let w_flat = w
                .reshape((
                    original_shape.dims()[0],
                    original_shape.dims()[1] * original_shape.dims()[2],
                ))
                .map_err(Either::Right)?;
            let ba_flat = scaled_ba
                .reshape((
                    original_shape.dims()[0],
                    original_shape.dims()[1] * original_shape.dims()[2],
                ))
                .map_err(Either::Right)?;

            // Compute row-wise norms of W + scaled_BA
            let w_plus_ba = (&w_flat + &ba_flat).map_err(Either::Right)?;
            let norms = w_plus_ba
                .sqr()
                .map_err(Either::Right)?
                .sum_keepdim(1) // Row-wise: sum over in_channels * kernel_size dimension
                .map_err(Either::Right)?
                .sqrt()
                .map_err(Either::Right)?
                .squeeze(1)
                .map_err(Either::Right)?;

            // Normalized weight: m * (W + scaled_BA) / ||W + scaled_BA||_2
            let norm_scale = m.broadcast_div(&norms).map_err(Either::Right)?;
            let normalized = w_plus_ba
                .broadcast_mul(&norm_scale)
                .map_err(Either::Right)?
                .reshape(original_shape.dims())
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
                FrozenConv1d::new(
                    &(self.old.weight() + self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias(),
                    *self.old.config(),
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
                FrozenConv1d::new(
                    &(self.old.weight() - self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias(),
                    *self.old.config(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraConv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.merged || self.scale.is_none() {
            self.old.forward(input)
        } else {
            let scale = self.scale.unwrap();

            // Dropout should only be applied to adapter path, not base model
            let input_adapter = if self.dropout.is_some() {
                self.dropout.as_ref().unwrap().forward(input, true)?
            } else {
                input.clone()
            };

            if let Some(ref m) = self.m {
                // DoRA implementation (memory-efficient, row-wise normalization)
                // Conv1d weights are [out_channels, in_channels, kernel_size]
                // DoRA: W' = m * (W + scale*BA) / ||W + scale*BA||_2
                // where ||·||_2 is row-wise L2 norm (norm across in_channels*kernel_size for each out_channel)
                // Per DoRA paper Section 3.1: magnitude is computed per output channel (row)
                // This mirrors loralinear.rs but adapted for Conv1d weight layout

                // Store bias to add at the end
                let bias = self.old.bias();

                // IMPORTANT: Compute W*x WITHOUT bias and WITHOUT dropout (bias added at the end)
                let w = self.old.weight();
                let w_conv = Conv1d::new(w.clone(), None, *self.config());
                let w_out = w_conv.forward(input)?; // Use original input

                // Compute adapter output: scale*B@A * x with dropout applied
                // For Conv1d, we need to apply B@A convolution
                let ba = self.b.matmul(&self.a)?.reshape(self.old.weight().shape())?;
                let ba_conv = Conv1d::new(ba.mul(scale)?, None, *self.config());
                let ba_out = ba_conv.forward(&input_adapter)?; // Use input with dropout

                // Combined output: W*x + scale*B@A*x (still no bias)
                let combined = (&w_out + &ba_out)?;

                // Compute row-wise norms of W + scale*BA efficiently
                // WITHOUT materializing the full BA matrix multiple times
                // For each row i: ||W_i + scale*BA_i||² = sum_j (W_ij + scale*(BA)_ij)²
                // We use: ||W_i + scale*BA_i||² = ||W_i||² + 2*scale*W_i·(BA)_i + scale²||BA_i||²

                let w = self.old.weight(); // [out_channels, in_channels, kernel_size]
                let original_shape = w.shape().clone();

                // Flatten to 2D for row-wise operations: [out_channels, in_channels * kernel_size]
                let w_flat = w.reshape((
                    original_shape.dims()[0],
                    original_shape.dims()[1] * original_shape.dims()[2],
                ))?;
                let b_weight = self.b.reshape((original_shape.dims()[0], self.b.dim(1)?))?;
                let a_weight = self.a.reshape((
                    self.a.dim(0)?,
                    original_shape.dims()[1] * original_shape.dims()[2],
                ))?;

                // Compute ||W_i||² for each row (sum over in_channels * kernel_size dimension)
                let w_norm_sq = w_flat.sqr()?.sum_keepdim(1)?; // [out_channels, 1]

                // Compute ||BA_i||² efficiently using the kernel trick
                // For each row i: ||BA_i||² = (BA)(BA)^T_ii = (B @ A @ A^T @ B^T)_ii
                let aat = a_weight.matmul(&a_weight.t()?)?; // [rank, rank]
                let b_aat = b_weight.matmul(&aat)?; // [out_channels, rank]
                let ba_norm_sq = (&b_aat * &b_weight)?.sum_keepdim(1)?; // [out_channels, 1]

                // Compute 2*W_i·(BA)_i efficiently
                let wa_t = w_flat.matmul(&a_weight.t()?)?; // [out_channels, rank]
                let cross = (&wa_t * &b_weight)?.sum_keepdim(1)?.mul(2.0 * scale)?; // [out_channels, 1]

                // ||W + scale*BA||² = ||W||² + 2*scale*W·BA + scale²||BA||²
                let norms = (w_norm_sq + cross + ba_norm_sq.mul(scale * scale)?)?
                    .sqrt()?
                    .squeeze(1)?; // [out_channels]

                // Apply DoRA row-wise: m * combined / norms
                // m is [out_channels], norms is [out_channels]
                // This normalizes each output channel (row) independently
                // For Conv1d, output is [batch, out_channels, seq_len], so reshape to [1, out_channels, 1]
                let norm_scale = m.broadcast_div(&norms)?; // [out_channels]
                let norm_scale = norm_scale.unsqueeze(0)?.unsqueeze(2)?; // [1, out_channels, 1]
                let mut result = combined.broadcast_mul(&norm_scale)?;

                // Add bias at the very end
                if let Some(b) = bias {
                    let b_reshaped = b.unsqueeze(0)?.unsqueeze(2)?; // [1, out_channels, 1]
                    result = result.broadcast_add(&b_reshaped)?;
                }
                Ok(result)
            } else {
                // Standard LoRA implementation
                let bias = self.bias().cloned();

                let mut weight = self.old.weight().clone();
                if self.dropout.is_some() {
                    weight = self
                        .dropout
                        .as_ref()
                        .unwrap()
                        .forward(&input_adapter, true)?;
                }

                let weight = (&weight
                    + &self
                        .b
                        .broadcast_matmul(&self.a.broadcast_matmul(&weight)?)?
                        .reshape(self.old.weight().shape())?
                        .mul(scale)?)?;

                let conv = Conv1d::new(weight, bias, *self.config());
                conv.forward(input)
            }
        }
    }
}

impl Saveable for LoraConv1d {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.a.clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.b.clone(),
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

impl Conv1dLayerLike for LoraConv1d {
    fn config(&self) -> &Conv1dConfig {
        self.old.config()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
}
