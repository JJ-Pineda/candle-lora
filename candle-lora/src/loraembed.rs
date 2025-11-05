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

        // For DoRA, apply magnitude normalization (column-wise)
        // Note: embeddings are stored as [num_embeddings, embedding_dim]
        // but we compute BA as [embedding_dim, num_embeddings]
        // We use column-wise normalization (per embedding vector)
        if let Some(ref m) = self.m {
            let w_t = self.embeddings().t().map_err(Either::Right)?;

            // Compute column-wise norms of W^T + scaled_BA efficiently
            // WITHOUT rematerializing BA multiple times
            // For each column j: ||W^T_j + scaled_BA_j||² = sum_i (W^T_ij + scaled_BA_ij)²
            // We use: ||W^T_j + scaled_BA_j||² = ||W^T_j||² + 2*scale*W^T_j·BA_j + scale²||BA_j||²

            let scale_val = self.scale.unwrap_or(1.0);

            // Compute ||W^T_j||² for each column (sum over embedding_dim dimension)
            let w_norm_sq = w_t
                .sqr()
                .map_err(Either::Right)?
                .sum_keepdim(0)
                .map_err(Either::Right)?; // [1, num_embeddings]

            // Compute ||BA_j||² efficiently using the kernel trick
            // For column j: ||BA_j||² = A_j^T @ B^T @ B @ A_j
            let btb = self
                .b
                .t()
                .map_err(Either::Right)?
                .matmul(&self.b)
                .map_err(Either::Right)?; // [rank, rank]
            let a_btb = self
                .a
                .t()
                .map_err(Either::Right)?
                .matmul(&btb)
                .map_err(Either::Right)?; // [num_embeddings, rank]
            let ba_norm_sq_cols = (&a_btb * &self.a.t().map_err(Either::Right)?)
                .map_err(Either::Right)?
                .sum_keepdim(1)
                .map_err(Either::Right)?
                .t()
                .map_err(Either::Right)?; // [1, num_embeddings]

            // Compute 2*W^T_j·BA_j efficiently
            // For each column: (B^T @ W^T) ⊙ A summed over rank dimension
            let bt_wt = self
                .b
                .t()
                .map_err(Either::Right)?
                .matmul(&w_t)
                .map_err(Either::Right)?; // [rank, num_embeddings]
            let cross = (&bt_wt * &self.a)
                .map_err(Either::Right)?
                .sum_keepdim(0)
                .map_err(Either::Right)?
                .mul(2.0 * scale_val)
                .map_err(Either::Right)?; // [1, num_embeddings]

            // ||W^T + scale*BA||² = ||W^T||² + 2*scale*W^T·BA + scale²||BA||²
            let norms = (w_norm_sq
                + cross
                + ba_norm_sq_cols
                    .mul(scale_val * scale_val)
                    .map_err(Either::Right)?)
            .map_err(Either::Right)?
            .sqrt()
            .map_err(Either::Right)?; // [1, num_embeddings]

            // Normalized weight: m * (W^T + scaled_BA) / ||W^T + scaled_BA||_2
            // Compute W^T + scaled_BA
            let w_plus_ba = (&w_t + &scaled_ba).map_err(Either::Right)?;

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
        if self.merged || self.scale.is_none() {
            self.old.forward(input)
        } else {
            let scale = self.scale.unwrap();
            if let Some(ref m) = self.m {
                // DoRA implementation (memory-efficient, column-wise normalization)
                // For embeddings: W^T is [embedding_dim, num_embeddings]
                // DoRA: W' = m * (W^T + scale*BA) / ||W^T + scale*BA||_2
                // where ||·||_2 is column-wise L2 norm (norm across embedding_dim for each token)
                // Per DoRA paper Section 3.1: magnitude is computed per embedding (column of W^T)
                // This mirrors loralinear.rs but adapted for embedding matrix layout

                // Note: Standard embedding lookup would give us W@x
                let result = self.old.forward(input)?;

                // Compute adapter contribution using the efficient embedding lookup
                let b = self.b.t()?;
                let b = b.reshape(b.shape())?;
                let after_a = self.embed_a.forward(input)?;
                let ba_out = after_a.broadcast_matmul(&b)?.mul(scale)?;

                // Combined output: W@x + scale*BA@x
                let combined = (&result + &ba_out)?;

                // Compute column-wise norms of W^T + scale*BA efficiently
                // WITHOUT materializing the full BA matrix
                // For each column j: ||W^T_j + scale*BA_j||² = sum_i (W^T_ij + scale*(BA)_ij)²
                // We use: ||W^T_j + scale*BA_j||² = ||W^T_j||² + 2*scale*W^T_j·(BA)_j + scale²||BA_j||²

                let w_t = self.embeddings().t()?; // [embedding_dim, num_embeddings]

                // Compute ||W^T_j||² for each column (sum over embedding_dim dimension)
                let w_norm_sq = w_t.sqr()?.sum_keepdim(0)?; // [1, num_embeddings]

                // Compute ||BA_j||² efficiently using the kernel trick
                // BA = [embedding_dim, num_embeddings]
                // For column j: ||BA_j||² = sum_i (BA)_ij² = sum_i (sum_k B_ik * A_kj)²
                // Using kernel trick: (BA_j)^T @ BA_j = A_j^T @ B^T @ B @ A_j
                let btb = self.b.t()?.matmul(&self.b)?; // [rank, rank]
                let a_btb = self.a.t()?.matmul(&btb)?; // [num_embeddings, rank]
                let ba_norm_sq_cols = (&a_btb * &self.a.t()?)?.sum_keepdim(1)?.t()?; // [1, num_embeddings]

                // Compute 2*W^T_j·(BA)_j efficiently
                // For each column j: W^T_j · (BA)_j = sum_i W^T_ij * (BA)_ij
                // where (BA)_ij = sum_k B_ik * A_kj
                // So: sum_i W^T_ij * sum_k B_ik * A_kj = sum_k (sum_i W^T_ij * B_ik) * A_kj
                //                                       = sum_k (B^T @ W^T)_kj * A_kj
                //                                       = (B^T @ W^T) ⊙ A summed over k for each column
                let bt_wt = self.b.t()?.matmul(&w_t)?; // [rank, num_embeddings]
                let cross = (&bt_wt * &self.a)?.sum_keepdim(0)?.mul(2.0 * scale)?; // [1, num_embeddings]

                // ||W^T + scale*BA||² = ||W^T||² + 2*scale*W^T·BA + scale²||BA||²
                let norms = (w_norm_sq + cross + ba_norm_sq_cols.mul(scale * scale)?)?
                    .sqrt()?
                    .squeeze(0)?; // [num_embeddings]

                // Apply DoRA column-wise: m * combined / norms
                // For embeddings, we need to apply the normalization per token
                // m is [embedding_dim, num_embeddings], norms is [num_embeddings]
                // We need to normalize each embedding vector (column of W^T)

                // Get norms for the input indices by indexing into our precomputed norms
                // Also index into m to get the magnitude values for selected embeddings
                let input_indices = input.flatten_all()?;
                let selected_norms = norms.index_select(&input_indices, 0)?;
                let selected_m = m.index_select(&input_indices, 1)?.t()?; // m is [embedding_dim, num_embeddings]

                // Normalize: result_normalized = (combined / norms) * m
                // selected_norms shape: [batch_flattened]
                // selected_m shape: [batch_flattened, embedding_dim]
                // combined shape: [batch_shape..., embedding_dim]

                let norm_scale = selected_m.broadcast_div(&selected_norms)?; // [batch_flattened, embedding_dim]

                // Reshape norm_scale to match combined's shape
                let combined_dims = combined.shape().dims();
                let mut target_shape = Vec::with_capacity(combined_dims.len());
                target_shape.extend_from_slice(&combined_dims[..combined_dims.len() - 1]);
                target_shape.push(self.hidden_size());

                let norm_scale_reshaped = norm_scale.reshape(target_shape.as_slice())?;
                Ok(combined.broadcast_mul(&norm_scale_reshaped)?)
            } else {
                // Standard LoRA implementation
                let mut result = self.old.forward(input)?;
                let b = self.b.t()?;
                let b = b.reshape(b.shape())?;

                let after_a = self.embed_a.forward(input)?;
                result = (result + (after_a.broadcast_matmul(&b)?).mul(scale))?;
                Ok(result)
            }
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
