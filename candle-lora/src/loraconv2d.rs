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

        // For DoRA, apply magnitude normalization
        if let Some(ref m) = self.m {
            let w = self.old.weight();

            // Compute column-wise norms of W + scaled_BA
            let w_plus_ba = (w + &scaled_ba).map_err(Either::Right)?;
            let norms = w_plus_ba
                .sqr()
                .map_err(Either::Right)?
                .sum_keepdim(0)
                .map_err(Either::Right)?
                .sqrt()
                .map_err(Either::Right)?;

            // Normalized weight: (m / norms) * (W + scaled_BA)
            let normalized = w_plus_ba
                .broadcast_div(&norms)
                .map_err(Either::Right)?
                .broadcast_mul(m)
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
            let mut a_input = input.clone();
            if self.dropout.is_some() {
                a_input = self.dropout.as_ref().unwrap().forward(input, true)?;
            }

            if let Some(ref m) = self.m {
                // DoRA implementation
                // Calculate W' using get_delta_weight logic
                let delta_weight = match self.old.weight().shape().dims()[2..4] {
                    [1, 1] => self
                        .b_conv
                        .weight()
                        .squeeze(3)?
                        .squeeze(2)?
                        .matmul(&self.a_conv.weight().squeeze(3)?.squeeze(2)?)?
                        .unsqueeze(2)?
                        .unsqueeze(3)?,
                    _ => {
                        let conv =
                            Conv2d::new(self.b_conv.weight().clone(), None, *self.old.config());
                        conv.forward(&self.a_conv.weight().permute((1, 0, 2, 3))?)?
                    }
                };

                let delta_weight = delta_weight.mul(scale)?;
                let w_prime = (self.old.weight() + delta_weight)?;

                // Calculate column-wise norm of W'
                let w_prime_norm = w_prime.sqr()?.sum_keepdim(0)?.sqrt()?;

                // Apply magnitude vector and normalize: m * W' / ||W'||
                let normalized_weight = w_prime.broadcast_div(&w_prime_norm)?;
                let scaled_weight = normalized_weight.broadcast_mul(m)?;

                // Apply the scaled weight
                let conv = Conv2d::new(scaled_weight, self.old.bias().cloned(), *self.old.config());
                conv.forward(input)
            } else {
                // Standard LoRA implementation
                let weight = self.old.forward(input)?;
                let tmp = self.b_conv.forward(&self.a_conv.forward(&a_input)?)?;
                &weight + tmp.mul(scale)?
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
