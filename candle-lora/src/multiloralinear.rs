use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, Linear, VarBuilder};
use either::Either;

use crate::{
    frozenlinear::FrozenLinear, LinearLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
    Saveable,
};

#[derive(Debug, Clone)]
pub struct MultiLoraLinear {
    old: Arc<FrozenLinear>,
    ff_a: HashMap<String, Linear>,
    ff_b: HashMap<String, Linear>,
    scale: Option<f64>,
    dropout: Option<Arc<Dropout>>,
    merged: bool,
    prefix: String,
    id: usize,
    m: HashMap<String, Tensor>, // DoRA magnitude vector
}
