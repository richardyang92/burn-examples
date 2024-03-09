
use burn::{backend::{Autodiff, LibTorch}, optim::AdamWConfig};
use training::TrainingConfig;

use crate::model::YoloV1Config;


mod model;
mod data;
mod training;

type MyBackend = LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    unsafe { backtrace_on_stack_overflow::enable() };
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(1);
    training::train::<MyAutodiffBackend>(
        "./yolo_v1/model",
        TrainingConfig::new(YoloV1Config::new(7, 2), AdamWConfig::new()),
        device,
    );
}
