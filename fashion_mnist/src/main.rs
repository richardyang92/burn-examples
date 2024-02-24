use burn::{backend::{wgpu::{AutoGraphicsApi, WgpuDevice}, Autodiff, Wgpu}, data::dataset::Dataset, optim::AdamConfig};

use crate::{data::FashionMNISTDataset, model::ModelConfig, training::TrainingConfig};

mod model;
mod data;
mod training;
mod inference;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::BestAvailable;
    training::train::<MyAutodiffBackend>(
        "./model",
        TrainingConfig::new(ModelConfig::new(10, 64 * 5 * 5), AdamConfig::new()),
        device,
    );
    // let dataset = FashionMNISTDataset::test();
    // let item = dataset.get(100).unwrap();
    // inference::infer::<MyAutodiffBackend>("./model", device, item);
}
