use std::usize;

use burn::{config::Config, module::Module, nn::{conv::{Conv2d, Conv2dConfig}, loss::CrossEntropyLossConfig, pool::{MaxPool2d, MaxPool2dConfig}, BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, ReLU}, tensor::{backend::{AutodiffBackend, Backend}, Int, Tensor}, train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep}};

use crate::data::FashionMNISTBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    batch_norm1: BatchNorm<B, 2>,
    pool1: MaxPool2d,
    conv2: Conv2d<B>,
    batch_norm2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    batch_norm3: BatchNorm<B, 2>,
    pool2: MaxPool2d,
    activation: ReLU,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 16], [5, 5])
                .with_padding(PaddingConfig2d::Explicit(2, 2)).init(device),
            batch_norm1: BatchNormConfig::new(16).init(device),
            pool1: MaxPool2dConfig::new([15, 15]).init(),
            conv2: Conv2dConfig::new([16, 32], [3, 3]).init(device),
            batch_norm2: BatchNormConfig::new(32).init(device),
            conv3: Conv2dConfig::new([32, 64], [3, 3]).init(device),
            batch_norm3: BatchNormConfig::new(64).init(device),
            pool2: MaxPool2dConfig::new([6, 6]).init(),
            activation: ReLU::new(),
            linear: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }

    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 16], [5, 5])
                .with_padding(PaddingConfig2d::Explicit(2, 2)).init_with(record.conv1),
            batch_norm1: BatchNormConfig::new(16).init_with(record.batch_norm1),
            pool1: MaxPool2dConfig::new([15, 15]).init(),
            conv2: Conv2dConfig::new([16, 32], [3, 3]).init_with(record.conv2),
            batch_norm2: BatchNormConfig::new(32).init_with(record.batch_norm2),
            conv3: Conv2dConfig::new([32, 64], [3, 3]).init_with(record.conv3),
            batch_norm3: BatchNormConfig::new(64).init_with(record.batch_norm3),
            pool2: MaxPool2dConfig::new([6, 6]).init(),
            activation: ReLU::new(),
            linear: LinearConfig::new(64 * 5 * 5, 10).init_with(record.linear),
        }
    }
}

impl<B: Backend> Model<B> {
    // N = (W − F + 2P ) / S + 1
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // conv 1 16x28x28
        let x = images.reshape([batch_size, 1, height, width]);
        let x = self.conv1.forward(x);
        let x = self.batch_norm1.forward(x);
        let x = self.activation.forward(x);

        // pool 1 16x14x14
        let x = self.pool1.forward(x);

        // conv 2 32x12x12
        let x = self.conv2.forward(x);
        let x = self.batch_norm2.forward(x);
        let x = self.activation.forward(x);

        // conv 3 64x10x10
        let x = self.conv3.forward(x);
        let x = self.batch_norm3.forward(x);
        let x = self.activation.forward(x);

        // pool 2 64x5x5
        let x = self.pool2.forward(x);

        let x = x.reshape([batch_size, 64 * 5 * 5]);

        // fc 10
        self.linear.forward(x)
    }

    pub fn forward_classification(&self, images: Tensor<B, 3>, targets: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new().init(&output.device()).forward(output.clone(), targets.clone());
        ClassificationOutput { loss, output, targets }
    }
}

impl<B: AutodiffBackend> TrainStep<FashionMNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FashionMNISTBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<FashionMNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FashionMNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}