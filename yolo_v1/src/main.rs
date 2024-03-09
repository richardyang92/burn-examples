
use burn::{backend::{Autodiff, LibTorch}, optim::AdamWConfig, tensor::Tensor};
// use data::{ItemLoader, VocItem, VocItemLoader};
use training::TrainingConfig;

use crate::model::YoloV1Config;


mod model;
mod data;
mod training;
mod inference;

type MyBackend = LibTorch;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    unsafe { backtrace_on_stack_overflow::enable() };
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    training::train::<MyAutodiffBackend>(
        "./yolo_v1/model",
        TrainingConfig::new(YoloV1Config::new(7, 2), AdamWConfig::new()),
        device,
    );

    // let data_loader = VocItemLoader {
    //     image_path: String::from("./yolo_v1/voc/JPEGImages/009961.jpg"),
    //     label_path: String::from("./yolo_v1/voc/Annotations/009961.xml"),
    // };

    // let voc_item = VocItem {
    //     image: data_loader.load_image().unwrap(),
    //     label: data_loader.load_lable().unwrap(),
    // };

    // inference::infer::<MyAutodiffBackend>("./yolo_v1/model", device, voc_item);
}
