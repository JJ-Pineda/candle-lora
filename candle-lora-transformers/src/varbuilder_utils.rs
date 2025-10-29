//! Utilities for creating a VarBuilder from a VarMap loaded from tensor storage formats.
use std::path::Path;

use candle_core::{DType, Device, Error, Var};
use candle_nn::{
    var_builder::{SimpleBackend, VarBuilderArgs},
    VarBuilder, VarMap,
};

use tqdm::Iter;

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
pub fn from_mmaped_safetensors<'a, P: AsRef<Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, Error> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };

        if silent {
            for (name, _) in tensors.tensors() {
                let verified_name = verify_name(&name);
                let tensor = tensors
                    .load(&name, device)?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(verified_name, Var::from_tensor(&tensor)?);
            }
        } else {
            for (name, _) in tensors.tensors().iter().tqdm() {
                let verified_name = verify_name(&name);
                let tensor = tensors
                    .load(&name, device)?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(verified_name, Var::from_tensor(&tensor)?);
            }
        };
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}

fn verify_name(name: &str) -> String {
    let name = if let Some(start) = name.find("model.layers.") {
        &name[start..]
    } else {
        name
    };
    //model.layers.0.mlp.up_proj.lora_A.weight
    let new_name = if let Some(bp) = name.strip_suffix(".lora_A.weight") {
        format!("{bp}.traced_lora_linear.a0.weight")
    } else if let Some(bp) = name.strip_suffix(".lora_B.weight") {
        format!("{bp}.traced_lora_linear.b0.weight")
    } else if let Some(bp) = name.strip_suffix(".lora_magnitude_vector") {
        format!("{bp}.traced_lora_linear.m0.weight")
    } else {
        name.to_string()
    };

    return new_name;
}

/// Load tensors into a VarBuilder backed by a VarMap using NpzTensors.
/// Set `silent` to not show a progress bar.
pub fn from_npz_tensors<'a, P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, Error> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        let tensors = candle_core::npy::NpzTensors::new(path)?;
        if silent {
            for name in tensors.names() {
                let tensor = tensors
                    .get(name)?
                    .expect("Expect Some(_) tensor.")
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(name.to_string(), Var::from_tensor(&tensor)?);
            }
        } else {
            for name in tensors.names().iter().tqdm() {
                let tensor = tensors
                    .get(name)?
                    .expect("Expect Some(_) tensor.")
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(name.to_string(), Var::from_tensor(&tensor)?);
            }
        };
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}

/// Load tensors into a VarBuilder backed by a VarMap using PthTensors.
/// Set `silent` to not show a progress bar.
pub fn from_pth_tensors<'a, P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>, Error> {
    let map = VarMap::new();
    {
        let mut ws = map.data().lock().unwrap();

        let tensors = candle_core::pickle::PthTensors::new(path, None)?;
        if silent {
            for name in tensors.tensor_infos().keys() {
                let tensor = tensors
                    .get(name)?
                    .expect("Tensor not found")
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(name.to_string(), Var::from_tensor(&tensor)?);
            }
        } else {
            for name in tensors.tensor_infos().keys().tqdm() {
                let tensor = tensors
                    .get(name)?
                    .expect("Tensor not found")
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(name.to_string(), Var::from_tensor(&tensor)?);
            }
        };
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}
