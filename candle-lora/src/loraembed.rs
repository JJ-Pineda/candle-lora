use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Embedding, Init, VarBuilder};
use either::Either;

use crate::{
    frozenembed::FrozenEmbedding, EmbeddingLayerLike, LoraConfig, Merge, MergeError,
    MergeErrorOrError, Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraEmbedding {
    old: Arc<FrozenEmbedding>,
    embed_a: Embedding,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    merged: bool,
    prefix: String,
    id: usize,
    m: Option<Tensor>,
}

#[derive(Clone, Debug)]
/// Configuration for LoraEmbedding, with `num_embeddings` vectors of `embedding_dim` size`.
pub struct LoraEmbeddingConfig {
    num_embeddings: usize,
    embedding_dim: usize,
}

impl LoraEmbeddingConfig {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        LoraEmbeddingConfig {
            num_embeddings,
            embedding_dim,
        }
    }
}

impl LoraEmbedding {
    pub fn new(
        old: &dyn EmbeddingLayerLike,
        embed_config: &LoraEmbeddingConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        let a = vb.pp(format!("a{id}")).get_with_hints(
            (config.rank, embed_config.num_embeddings),
            "weight",
            init::ZERO,
        )?;
        let b: Tensor = vb.pp(format!("b{id}")).get_with_hints(
            (embed_config.embedding_dim, config.rank),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        let mut a_t = a.t()?;
        a_t = a_t.reshape(a_t.shape())?;
        let embed_a = Embedding::new(a_t.clone(), a_t.dim(1)?);

        // Try to load magnitude vector for DoRA
        let m: Option<Tensor> = if vb.contains_tensor(&format!("m{id}.weight")) {
            vb.pp(format!("m{id}"))
                .get(
                    (embed_config.embedding_dim, embed_config.num_embeddings),
                    "weight",
                )
                .ok()
        } else {
            None
        };

        Ok(LoraEmbedding {
            old: Arc::new(FrozenEmbedding::new_from_embed(old)?),
            embed_a,
            a,
            b,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            merged: false,
            prefix: vb.prefix(),
            id,
            m,
        })
    }
}

impl Merge for LoraEmbedding {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let ba = self.b.matmul(&self.a).map_err(Either::Right)?;

        let scaled_ba = match self.scale {
            Some(scale) => ba.mul(scale).map_err(Either::Right)?,
            None => ba,
        };

        // For DoRA, apply magnitude normalization
        // Note: embeddings are stored as [num_embeddings, embedding_dim]
        // but we compute BA as [embedding_dim, num_embeddings]
        if let Some(ref m) = self.m {
            // Compute W^T + scaled_BA (both are [embedding_dim, num_embeddings])
            let w_t = self.embeddings().t().map_err(Either::Right)?;
            let w_plus_ba = (&w_t + &scaled_ba).map_err(Either::Right)?;

            // Compute column-wise norms
            let norms = w_plus_ba
                .sqr()
                .map_err(Either::Right)?
                .sum_keepdim(0)
                .map_err(Either::Right)?
                .sqrt()
                .map_err(Either::Right)?;

            // Normalized weight: (m / norms) * (W^T + scaled_BA)
            let normalized = w_plus_ba
                .broadcast_div(&norms)
                .map_err(Either::Right)?
                .broadcast_mul(m)
                .map_err(Either::Right)?;

            // Delta is (normalized - W^T), return as is (will be transposed by caller)
            Ok((normalized - &w_t).map_err(Either::Right)?)
        } else {
            // Standard LoRA - return BA
            Ok(scaled_ba)
        }
    }

    fn merge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if self.merged {
            Err(Either::Left(MergeError::AlreadyMerged))
        } else {
            self.old = Arc::new(
                FrozenEmbedding::new(
                    &(self.embeddings() + self.get_delta_weight()?.transpose(0, 1))
                        .map_err(Either::Right)?,
                    self.hidden_size(),
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
                FrozenEmbedding::new(
                    &(self.embeddings() - self.get_delta_weight()?.transpose(0, 1))
                        .map_err(Either::Right)?,
                    self.hidden_size(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraEmbedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if let Some(ref m) = self.m {
            // DoRA implementation
            if let Some(scale) = self.scale {
                // Calculate W' = W + BA * scale
                let delta = self.b.matmul(&self.a)?.mul(scale)?;
                let w_prime = (self.embeddings().t()? + delta)?;

                // Calculate column-wise norm of W'
                let w_prime_norm = w_prime.sqr()?.sum_keepdim(0)?.sqrt()?;

                // Apply magnitude vector and normalize: m * W' / ||W'||
                let normalized_weight = w_prime.broadcast_div(&w_prime_norm)?;
                let scaled_weight = normalized_weight.broadcast_mul(m)?;

                // Transpose back and create embedding (ensure contiguous)
                let embedding_weight = scaled_weight.t()?.contiguous()?;
                let embed = Embedding::new(embedding_weight, self.hidden_size());
                embed.forward(input)
            } else {
                self.old.forward(input)
            }
        } else {
            // Standard LoRA implementation
            let mut result = self.old.forward(input)?;
            if let Some(scale) = self.scale {
                let b = self.b.t()?;
                let b = b.reshape(b.shape())?;

                let after_a = self.embed_a.forward(input)?;
                result = (result + (after_a.broadcast_matmul(&b)?).mul(scale))?
            }
            Ok(result)
        }
    }
}

impl Saveable for LoraEmbedding {
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

impl EmbeddingLayerLike for LoraEmbedding {
    fn embeddings(&self) -> &Tensor {
        self.old.embeddings()
    }
    fn hidden_size(&self) -> usize {
        self.old.hidden_size()
    }
}
