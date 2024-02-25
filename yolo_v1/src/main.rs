use burn::{backend::{wgpu::{AutoGraphicsApi, WgpuDevice}, Wgpu}, tensor::{Data, Tensor}};
use data::VocItemLoader;

use crate::{data::ItemLoader, model::YoloV1Config};

mod model;
mod data;

type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;

fn main() {
    let device = WgpuDevice::BestAvailable;

    let yolo_v1 = YoloV1Config::new(7, 2).init::<MyBackend>(&device);

    let item_loader = VocItemLoader {
        image_path: String::from("/media/yang/MyFiles/Projects/burn-examples/yolo_v1/voc/JPEGImages/000005.jpg"),
        label_path: String::from("/media/yang/MyFiles/Projects/burn-examples/yolo_v1/voc/Annotations/000005.xml"),
    };

    let image = item_loader.load_image().unwrap();
    // let label = item_loader.load_lable();
    let input_tensor = Tensor::<MyBackend, 3>::from(Data::<u8, 3>::from(image).convert()).reshape([1, 3, 448, 448]);
    let output_tennsor = yolo_v1.forward(input_tensor);
    println!("{:?}", output_tennsor.dims());
}
