use burn::{config::Config, record::{CompactRecorder, Recorder}, tensor::{activation::softmax, backend::Backend, Data, Int, Tensor}};
use rand::Rng;

use crate::{tokenizer::{SimpleTokenizerConfig, Tokenizer}, training::TrainingConfig};

const TINY_SHAKESPEARE: &str = "bigram/data/tinyshakespeare.txt";
const TOKENIZER_VOCAB: &str = "bigram/data/tokenizer.txt";

pub fn generate<B: Backend>(artifact_dir: &str, device: B::Device, input: &str, max_new_token: usize) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
    .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record);
    let simple_tokenizer = SimpleTokenizerConfig::new(
        String::from(TINY_SHAKESPEARE),
        String::from(TOKENIZER_VOCAB))
        .init().expect("tokenizer config file should exist!");

    let encoded_str = simple_tokenizer.encode(input);
    let context = encoded_str.as_slice().into_iter().map(|item| *item as i16).collect::<Vec<i16>>();
    let mut input = Tensor::<B, 1, Int>::from_data(Data::from(context.as_slice()).convert(), &device);

    for _ in 0..max_new_token {
        // println!("input: {:?}", input.to_data().value);
        let [input_dim] = input.dims();
        let logits = model.forward(input.clone().reshape([1, input_dim]));
        let [b, t, c] = logits.dims();
        let probs: Tensor<B, 2> = softmax(logits.slice::<3>([0..b, t-1..t, 0..c]).squeeze(1), 1);
        let prob_elems = probs.to_data().convert::<f32>().value;
        // println!("prob_elems: {:?}", prob_elems.len());
        let elem_next = sample_distribution(&prob_elems);
        // println!("elem_next={:?}", elem_next);
        let input_next = Tensor::<B, 1, Int>::from(Data::from([elem_next as i16]).convert()).to_device(&device);
        input = Tensor::cat(vec![input.clone(), input_next], 0);
    }

    println!("{:?}", simple_tokenizer.decode(&input.to_data().convert::<i16>().value.as_slice()
        .into_iter().map(|item| *item as usize).collect::<Vec<usize>>()));
}

fn sample_distribution(distribution: &[f32]) -> usize {
    let mut cdf = Vec::with_capacity(distribution.len());  
    let mut sum = 0.0;
    for &prob in distribution.iter() {
        sum += prob;
        cdf.push(sum);
    }  
  
    // Normalize the CDF if necessary  
    let cdf_last = *cdf.last().unwrap();  
    if cdf_last != 1.0 {  
        for cdf_val in cdf.iter_mut() {  
            *cdf_val /= cdf_last;  
        }  
    }  
  
    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(0f32..1f32);
  
    // Step 4: Find the index in the CDF  
    cdf.iter().position(|&x| x >= random_value).unwrap_or_else(|| cdf.len() - 1)
}