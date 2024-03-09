
use burn::{config::Config, data::{dataloader::DataLoaderBuilder, dataset::Dataset}, module::Module, optim::AdamWConfig, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{metric::{store::{Aggregate, Direction, Split}, CpuMemory, CpuUse, LossMetric}, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition}};

use crate::{data::{YoloV1Batcher, VocDataset}, model::YoloV1Config};

// const VOC2007_ROOT: &'static str = "/Users/yangyang/Projects/burn-examples/yolo_v1/data";
const VOC2007_ROOT: &'static str = "/media/yang/MyFiles/VOC2007";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: YoloV1Config,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 35)]
    pub seed: u64,
    #[config(default = 0.001)]
    pub learing_rate: f64,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = YoloV1Batcher::<B>::new(device.clone());
    let batcher_valid = YoloV1Batcher::<B::InnerBackend>::new(device.clone());

    let mut dataset_total = VocDataset::new(VOC2007_ROOT);
    dataset_total.shuffle_with_seed(config.seed);
    let (dataset_train, dataset_val) = dataset_total.split_by_ratio(7);

    println!("Train Dataset Size: {}", dataset_train.len());
    println!("Test Dataset Size: {}", dataset_val.len());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_val);

    let learner = LearnerBuilder::new(artifact_dir)
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .metric_train_numeric(CpuUse::new())
    .metric_valid_numeric(CpuUse::new())
    .metric_train_numeric(CpuMemory::new())
    .metric_valid_numeric(CpuMemory::new())
    .with_file_checkpointer(CompactRecorder::new())
    .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
        Aggregate::Mean,
        Direction::Lowest,
        Split::Valid,
        StoppingCondition::NoImprovementSince { n_epochs: 1 },
    ))
    .devices(vec![device.clone()])
    .num_epochs(config.num_epochs)
    .build(config.model.init(&device), config.optimizer.init(), config.learing_rate);
    
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}