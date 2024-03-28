use std::env;

use burn::{backend::{wgpu::{AutoGraphicsApi, WgpuDevice}, Autodiff, Wgpu}, data::dataset::Dataset, optim::AdamConfig};

use crate::{data::FashionMNISTDataset, model::ModelConfig, training::TrainingConfig};

mod model;
mod data;
mod training;
mod inference;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let args = env::args().collect::<Vec<String>>();

    let device = WgpuDevice::BestAvailable;

    if args[1].eq("train") {
        training::train::<MyAutodiffBackend>(
            "./fashion_mnist/model",
            TrainingConfig::new(ModelConfig::new(10, 64 * 5 * 5), AdamConfig::new()),
            device,
        );
    } else {
        let dataset = FashionMNISTDataset::test();
        let item = dataset.get(100).unwrap();
        inference::infer::<MyAutodiffBackend>("./model", device, item);
    }
}
