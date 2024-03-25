use std::env;

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
    let args: Vec<String> = env::args().collect();
    if args[1].eq("train") {
        training::train::<MyAutodiffBackend>(
            "./bigram/model",
            TrainingConfig::new(BigramLanageModelConfig::new(), AdamWConfig::new()),
            device,
        );
    } else if args[1].eq("generate") {
        inference::generate::<MyAutodiffBackend>("./bigram/model", device, " ", 500);
    }
}
