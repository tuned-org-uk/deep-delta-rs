//! Binary entrypoint for the deep-delta-learn workspace.
//!
//! The goal of this binary is to provide a quick smoke-test that:
//! - the selected Burn backend initializes,
//! - the core Delta operator runs,
//! - the DeltaResidual block runs and returns tensors of expected shapes.

use burn::prelude::*;
use burn::tensor::Distribution;

use deep_delta_learn::{delta::delta_update, nn::DeltaResidualConfig};

use deep_delta_learn::backend::{get_device, print_backend_info, AutoBackend};

type B = AutoBackend;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    print_backend_info();
    let device = get_device();

    // ------------------------------------------------------------------
    // 1) Smoke test: delta_update
    // ------------------------------------------------------------------
    let x = Tensor::<B, 3>::random([2, 8, 4], Distribution::Uniform(-1.0, 1.0), &device);
    let k = Tensor::<B, 2>::random([2, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let v = Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);
    let beta = Tensor::<B, 2>::random([2, 1], Distribution::Uniform(0.0, 2.0), &device);

    let y = delta_update(x.clone(), k, v, beta, 1e-8);
    println!("delta_update: x {:?} -> y {:?}", x.dims(), y.dims());

    // ------------------------------------------------------------------
    // 2) Smoke test: DeltaResidual (internal branches)
    // ------------------------------------------------------------------
    let cfg = DeltaResidualConfig::new(8, 4, 16).with_eps(1e-8);
    let block = cfg.init::<B>(&device);

    let (y2, k2, beta2, v2) = block.forward_with_params(x);

    println!("DeltaResidual: y  {:?}", y2.dims());
    println!("  k     {:?}", k2.dims());
    println!("  beta  {:?}", beta2.dims());
    println!("  v     {:?}", v2.dims());

    Ok(())
}
