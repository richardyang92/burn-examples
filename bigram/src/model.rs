use burn::{config::Config, module::Module, nn::{loss::CrossEntropyLossConfig, Embedding, EmbeddingConfig}, tensor::{backend::{AutodiffBackend, Backend}, Int, Tensor}, train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep}};


use crate::data::BigramBatch;

#[derive(Config)]
pub struct BigramLanageModelConfig {
    #[config(default = 65)]
    vocab_size: usize,
    #[config(default = 65)]
    d_model: usize,
}

impl BigramLanageModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramLanageModel<B> {
        BigramLanageModel {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device)
        }
    }

    pub fn init_with<B: Backend>(&self, record: BigramLanageModelRecord<B>) -> BigramLanageModel<B> {
        BigramLanageModel {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init_with(record.embedding)
        }
    }
}

#[derive(Module, Debug)]
pub struct BigramLanageModel<B: Backend> {
    embedding: Embedding<B>,
}

impl<B: Backend> BigramLanageModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(input)
    }

    pub fn forward_classification(&self, input: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> ClassificationOutput<B> {
        let output = self.forward(input);
        let [b, t, c] = output.dims();
        let [b_prim, t_prim] = targets.dims();
        let output = output.reshape([b * t, c]);
        let targets = targets.reshape([b_prim * t_prim]);
        let loss = CrossEntropyLossConfig::new().init(&output.device()).forward(output.clone(), targets.clone());
        ClassificationOutput { loss, output, targets }
    }
}

impl<B: AutodiffBackend> TrainStep<BigramBatch<B>, ClassificationOutput<B>> for BigramLanageModel<B> {
    fn step(&self, batch: BigramBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<BigramBatch<B>, ClassificationOutput<B>> for BigramLanageModel<B> {
    fn step(&self, batch: BigramBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}