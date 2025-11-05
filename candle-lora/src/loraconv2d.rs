use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Conv2d, Conv2dConfig, Dropout, VarBuilder};
use either::Either;

use crate::{
    frozenconv::FrozenConv2d, Conv2dLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
    Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraConv2d {
    old: Arc<FrozenConv2d>,
    a_conv: Conv2d,
    b_conv: Conv2d,
    scale: Option<f64>,
    dropout: Option<Arc<Dropout>>,
    merged: bool,
    prefix: String,
    id: usize,
    m: Option<Tensor>,
}

#[derive(Clone, Debug)]
/// Configuration for LoraConv2d. Other configurations are inherited from the `Conv2d` struct.
pub struct LoraConv2dConfig {
    in_channels: usize,
    out_channels: usize,
}

impl LoraConv2dConfig {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        LoraConv2dConfig {
            in_channels,
            out_channels,
        }
    }
}

impl LoraConv2d {
    pub fn new(
        old: &dyn Conv2dLayerLike,
        conv_config: &LoraConv2dConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        let a = vb.pp(format!("a{id}")).get_with_hints(
            (
                config.rank,
                conv_config.in_channels / old.config().groups,
                old.weight().dim(2).unwrap(),
                old.weight().dim(3).unwrap(),
            ),
            "weight",
            init::DEFAULT_KAIMING_NORMAL,
        )?;
        let b = vb.pp(format!("b{id}")).get_with_hints(
            (
                conv_config.out_channels,
                config.rank / old.config().groups,
                1,
                1,
            ),
            "weight",
            init::ZERO,
        )?;

        let a_conv = Conv2d::new(a, None, *old.config());
        let b_conv = Conv2d::new(
            b,
            None,
            Conv2dConfig {
                stride: 1,
                ..*old.config()
            },
        );

        // Try to load magnitude vector for DoRA
        let m: Option<Tensor> = if vb.contains_tensor(&format!("m{id}.weight")) {
            vb.pp(format!("m{id}"))
                .get(
                    (
                        conv_config.out_channels,
                        conv_config.in_channels / old.config().groups,
                        old.weight().dim(2).unwrap(),
                        old.weight().dim(3).unwrap(),
                    ),
                    "weight",
                )
                .ok()
        } else {
            None
        };

        Ok(LoraConv2d {
            old: Arc::new(FrozenConv2d::new_from_conv2d(old)?),
            a_conv,
            b_conv,
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

impl Merge for LoraConv2d {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let ba = match self.old.weight().shape().dims()[2..4] {
            [1, 1] => self
                .b_conv
                .weight()
                .squeeze(3)
                .map_err(Either::Right)?
                .squeeze(2)
                .map_err(Either::Right)?
                .matmul(
                    &self
                        .a_conv
                        .weight()
                        .squeeze(3)
                        .map_err(Either::Right)?
                        .squeeze(2)
                        .map_err(Either::Right)?,
                )
                .map_err(Either::Right)?
                .unsqueeze(2)
                .map_err(Either::Right)?
                .unsqueeze(3)
                .map_err(Either::Right)?,
            _ => {
                let conv = Conv2d::new(self.b_conv.weight().clone(), None, *self.old.config());
                conv.forward(
                    &self
                        .a_conv
                        .weight()
                        .permute((1, 0, 2, 3))
                        .map_err(Either::Right)?,
                )
                .map_err(Either::Right)?
            }
        };

        let scaled_ba = match self.scale {
            Some(scale) => ba.mul(scale).map_err(Either::Right)?,
            None => ba,
        };

        // For DoRA, apply magnitude normalization (row-wise)
        // Conv2d weights are [out_channels, in_channels, kernel_h, kernel_w]
        // We flatten to [out_channels, in_channels * kernel_h * kernel_w] for row-wise normalization
        if let Some(ref m) = self.m {
            let w = self.old.weight();
            let original_shape = w.shape().clone();

            // Flatten weight to 2D: [out_channels, in_channels * kernel_h * kernel_w]
            let out_channels = original_shape.dims()[0];
            let spatial_dim =
                original_shape.dims()[1] * original_shape.dims()[2] * original_shape.dims()[3];
            let w_flat = w
                .reshape((out_channels, spatial_dim))
                .map_err(Either::Right)?;
            let ba_flat = scaled_ba
                .reshape((out_channels, spatial_dim))
                .map_err(Either::Right)?;

            // Compute row-wise norms of W + scaled_BA
            let w_plus_ba = (&w_flat + &ba_flat).map_err(Either::Right)?;
            let norms = w_plus_ba
                .sqr()
                .map_err(Either::Right)?
                .sum_keepdim(1) // Row-wise: sum over in_channels * kernel_h * kernel_w dimension
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
                FrozenConv2d::new(
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
                FrozenConv2d::new(
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

impl Module for LoraConv2d {
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
                // Conv2d weights are [out_channels, in_channels, kernel_h, kernel_w]
                // DoRA: W' = m * (W + scale*BA) / ||W + scale*BA||_2
                // where ||·||_2 is row-wise L2 norm (norm across in_channels*kernel_h*kernel_w for each out_channel)
                // Per DoRA paper Section 3.1: magnitude is computed per output channel (row)
                // This mirrors loralinear.rs but adapted for Conv2d weight layout

                // Store bias to add at the end
                let bias = self.old.bias();

                // IMPORTANT: Compute W*x WITHOUT bias and WITHOUT dropout (bias added at the end)
                let w = self.old.weight();
                let w_conv = Conv2d::new(w.clone(), None, *self.config());
                let w_out = w_conv.forward(input)?; // Use original input

                // Compute adapter output: scale*B@A * x with dropout applied
                let ba_out = self
                    .b_conv
                    .forward(&self.a_conv.forward(&input_adapter)?)? // Use input with dropout
                    .mul(scale)?;

                // Combined output: W*x + scale*B@A*x (still no bias)
                let combined = (&w_out + &ba_out)?;

                // Compute row-wise norms of W + scale*BA efficiently
                // WITHOUT materializing the full BA matrix multiple times
                // For each row i: ||W_i + scale*BA_i||² = sum_j (W_ij + scale*(BA)_ij)²
                // We use: ||W_i + scale*BA_i||² = ||W_i||² + 2*scale*W_i·(BA)_i + scale²||BA_i||²

                let w = self.old.weight(); // [out_channels, in_channels, kernel_h, kernel_w]
                let original_shape = w.shape().clone();

                // Flatten to 2D for row-wise operations: [out_channels, in_channels * kernel_h * kernel_w]
                let out_channels = original_shape.dims()[0];
                let spatial_dim =
                    original_shape.dims()[1] * original_shape.dims()[2] * original_shape.dims()[3];
                let w_flat = w.reshape((out_channels, spatial_dim))?;

                // Get B and A weights and flatten appropriately
                let b_weight = self.b_conv.weight(); // [out_channels, rank, 1, 1]
                let a_weight = self.a_conv.weight(); // [rank, in_channels, kernel_h, kernel_w]

                let b_flat = b_weight.reshape((out_channels, b_weight.dim(1)?))?; // [out_channels, rank]
                let a_flat = a_weight.reshape((a_weight.dim(0)?, spatial_dim))?; // [rank, in_channels * kernel_h * kernel_w]

                // Compute ||W_i||² for each row (sum over spatial dimension)
                let w_norm_sq = w_flat.sqr()?.sum_keepdim(1)?; // [out_channels, 1]

                // Compute ||BA_i||² efficiently using the kernel trick
                // For each row i: ||BA_i||² = (BA)(BA)^T_ii = (B @ A @ A^T @ B^T)_ii
                let aat = a_flat.matmul(&a_flat.t()?)?; // [rank, rank]
                let b_aat = b_flat.matmul(&aat)?; // [out_channels, rank]
                let ba_norm_sq = (&b_aat * &b_flat)?.sum_keepdim(1)?; // [out_channels, 1]

                // Compute 2*W_i·(BA)_i efficiently
                let wa_t = w_flat.matmul(&a_flat.t()?)?; // [out_channels, rank]
                let cross = (&wa_t * &b_flat)?.sum_keepdim(1)?.mul(2.0 * scale)?; // [out_channels, 1]

                // ||W + scale*BA||² = ||W||² + 2*scale*W·BA + scale²||BA||²
                let norms = (w_norm_sq + cross + ba_norm_sq.mul(scale * scale)?)?
                    .sqrt()?
                    .squeeze(1)?; // [out_channels]

                // Apply DoRA row-wise: m * combined / norms
                // m is [out_channels], norms is [out_channels]
                // This normalizes each output channel (row) independently
                // For Conv2d, output is [batch, out_channels, height, width], so reshape to [1, out_channels, 1, 1]
                let norm_scale = m.broadcast_div(&norms)?; // [out_channels]
                let norm_scale = norm_scale.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?; // [1, out_channels, 1, 1]
                let mut result = combined.broadcast_mul(&norm_scale)?;

                // Add bias at the very end
                if let Some(b) = bias {
                    let b_reshaped = b.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?; // [1, out_channels, 1, 1]
                    result = result.broadcast_add(&b_reshaped)?;
                }
                Ok(result)
            } else {
                // Standard LoRA implementation
                // Base model uses original input (no dropout)
                let mut result = self.old.forward(input)?;
                // Adapter uses input with dropout
                let adapter_out = self
                    .b_conv
                    .forward(&self.a_conv.forward(&input_adapter)?)?
                    .mul(scale)?;
                result = (result + adapter_out)?;
                Ok(result)
            }
        }
    }
}

impl Saveable for LoraConv2d {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.a_conv.weight().clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.b_conv.weight().clone(),
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

impl Conv2dLayerLike for LoraConv2d {
    fn config(&self) -> &Conv2dConfig {
        self.old.config()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
}
