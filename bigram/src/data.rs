use std::{fs::File, io::Read};

use burn::{data::{dataloader::batcher::Batcher, dataset::Dataset}, tensor::{backend::Backend, Data, Int, Tensor}};
use derive_new::new;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::tokenizer::{SimpleTokenizerConfig, Tokenizer};

#[derive(Debug, new, Clone)]
pub struct TinyShakespeareItem {
    context: Vec<i16>,
    target: Vec<i16>,
}

pub struct TinyShakespeareDataset {
    block_size: usize,
    content: Vec<i16>,
}

impl TinyShakespeareDataset {
    pub fn new(dataset: &str, vocab: &str, split: &str) -> Self {
        let simple_tokenizer = SimpleTokenizerConfig::new(
            String::from(dataset),
            String::from(vocab))
            .init().expect("tokenizer config file should exist!");
        let mut tinyshakespeare = File::open(dataset).unwrap();
        let mut tinyshakespeare_contents = String::new();
        tinyshakespeare.read_to_string(&mut tinyshakespeare_contents)
            .expect(format!("read contents from {} should success", dataset).as_str());

        let tinyshakespeare_slice = tinyshakespeare_contents.as_str();
        let tinyshakespeare_len = tinyshakespeare_slice.len();
        let split_idx = tinyshakespeare_len * 7 / 10;
        let tinyshakespeare_slice = if split.eq("train") {
            &tinyshakespeare_slice[0..split_idx]
        } else {
            &tinyshakespeare_slice[split_idx + 1..tinyshakespeare_len]
        };

        let content = simple_tokenizer.encode(tinyshakespeare_slice)
            .par_iter().map(|item| *item as i16).collect::<Vec<i16>>();

        Self { 
            block_size: 8,
            content
        }
    }
}

impl Dataset<TinyShakespeareItem> for TinyShakespeareDataset {
    fn get(&self, index: usize) -> Option<TinyShakespeareItem> {
        if index > self.content.len() - self.block_size - 1 {
            return None;
        }
        let context = self.content.as_slice()[index..index + self.block_size].to_vec();
        let target = self.content.as_slice()[index + 1..index + self.block_size + 1].to_vec();
        Some(TinyShakespeareItem {
            context, target
        })
    }

    fn len(&self) -> usize {
        self.content.len() - self.block_size
    }
}

#[derive(new)]
pub struct BigramBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Debug, Clone)]
pub struct BigramBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<TinyShakespeareItem, BigramBatch<B>> for BigramBatcher<B> {
    fn batch(&self, items: Vec<TinyShakespeareItem>) -> BigramBatch<B> {
        let inputs = items.par_iter()
            .map(|item| &item.context)
            .map(|context| Data::from(context.as_slice()))
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert(), &self.device))
            .collect::<Vec<Tensor<B, 1, Int>>>();
        let inputs = Tensor::stack(inputs, 0);

        let targets = items.par_iter()
            .map(|item| &item.target)
            .map(|context| Data::from(context.as_slice()))
            .map(|data| Tensor::from_data(data.convert(), &self.device))
            .collect::<Vec<Tensor<B, 1, Int>>>();
        let targets = Tensor::stack(targets, 0);

        BigramBatch {
            inputs,
            targets,
        }
    }
}