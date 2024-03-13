use burn::{backend::{libtorch::LibTorchDevice, Autodiff, LibTorch}, optim::AdamWConfig};
use model::BigramLanageModelConfig;
use training::TrainingConfig;

mod tokenizer;
mod model;
mod data;
mod training;
mod inference;

type MyBackend = LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = LibTorchDevice::Mps;
    // training::train::<MyAutodiffBackend>(
    //     "./bigram/model",
    //     TrainingConfig::new(BigramLanageModelConfig::new(), AdamWConfig::new()),
    //     device,
    // );

    inference::generate::<MyAutodiffBackend>("./bigram/model", device, " ", 500);
}
