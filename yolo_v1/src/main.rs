
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
    let device = burn::backend::libtorch::LibTorchDevice::Mps;
    training::train::<MyAutodiffBackend>(
        "./yolo_v1/model",
        TrainingConfig::new(YoloV1Config::new(7, 2), AdamWConfig::new()),
        device,
    );
    // let dataloader = VocItemLoader {
    //     image_path: String::from("/Users/yangyang/Projects/burn-examples/yolo_v1/data/JPEGImages/009935.jpg"),
    //     label_path: String::from("/Users/yangyang/Projects/burn-examples/yolo_v1/data/Annotations/009935.xml"),
    // };
    // let image = dataloader.load_image().unwrap();
    // let label = dataloader.load_lable().unwrap();

    // let image_tensor = Tensor::<MyBackend, 1>::from_data(Data::from(image.as_slice()).convert(), &device).reshape([1, 3, 448, 448]);
    // let label_tensor = Tensor::<MyBackend, 1>::from_data(Data::from(label.as_slice()), &device).reshape([1, 30, 7, 7]);
    // println!("image_tensor: {:?}", image_tensor.dims());
    // println!("label_tensor: {:?}", label_tensor.into_data());

    // let dataloader_2 = VocItemLoader {
    //     image_path: String::from("/Users/yangyang/Projects/burn-examples/yolo_v1/data/JPEGImages/009938.jpg"),
    //     label_path: String::from("/Users/yangyang/Projects/burn-examples/yolo_v1/data/Annotations/009938.xml"),
    // };

    // let image_2 = dataloader_2.load_image().unwrap();
    // let label_2 = dataloader_2.load_lable().unwrap();

    // let image_tensor_2 = Tensor::<MyBackend, 1>::from_data(Data::from(image_2.as_slice()).convert(), &device).reshape([1, 3, 448, 448]);
    // let label_tensor_2 = Tensor::<MyBackend, 3>::from_data(Data::<f32, 3>::from(label_2), &device).reshape([1, 30, 7, 7]);
    // println!("image_tensor: {:?}", image_tensor_2.dims());
    // println!("label_tensor: {:?}", label_tensor_2.dims());

    // let image = Tensor::cat(vec![image_tensor, image_tensor_2], 0);
    // let label = Tensor::cat(vec![label_tensor, label_tensor_2], 0);

    // let yolo_v1 = YoloV1Config::new(7, 2).init::<MyBackend>(&device);
    // let output = yolo_v1.forward_regression(image, label);
    // println!("loss={:#}", output);
}
