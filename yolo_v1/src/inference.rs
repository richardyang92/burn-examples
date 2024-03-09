use burn::{config::Config, data::dataloader::batcher::Batcher, record::{CompactRecorder, Recorder}, tensor::backend::Backend};

use crate::{data::{VocItem, YoloV1Batcher}, training::TrainingConfig};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: VocItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record);

    let label = item.clone().label;
    let batcher = YoloV1Batcher::<B>::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);

    println!("Predicted {:?} Expected {:?}", output.squeeze::<3>(0).to_data().value, label);
}