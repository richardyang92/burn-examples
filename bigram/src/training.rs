use burn::{config::Config, data::dataloader::DataLoaderBuilder, module::Module, optim::AdamWConfig, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{metric::{store::{Aggregate, Direction, Split}, AccuracyMetric, CpuMemory, CpuUse, LossMetric}, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition}};

use crate::{data::{BigramBatcher, TinyShakespeareDataset}, model::BigramLanageModelConfig};

const TINY_SHAKESPEARE: &str = "bigram/data/tinyshakespeare.txt";
const TOKENIZER_VOCAB: &str = "bigram/data/tokenizer.txt";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: BigramLanageModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learing_rate: f64,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = BigramBatcher::<B>::new(device.clone());
    let batcher_valid = BigramBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TinyShakespeareDataset::new(TINY_SHAKESPEARE, TOKENIZER_VOCAB, "train"));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TinyShakespeareDataset::new(TINY_SHAKESPEARE, TOKENIZER_VOCAB, "val"));

        let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(config.model.init(&device), config.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}