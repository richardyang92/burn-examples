use std::{fs::File, io::{Error, ErrorKind, Read, Result, Write}, path::Path};

use derive_new::new;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, value: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}

#[derive(new)]
pub struct SimpleTokenizerConfig {
    dataset: String,
    tokenizer_config: String,
}

impl SimpleTokenizerConfig {
    pub fn init(&self) -> Option<SimpleTokenizer> {
        let config_filepath = Path::new(&self.tokenizer_config);
        let config_file: Result<File> = if !config_filepath.exists() {
            let mut token_file = File::open(&self.dataset).unwrap();
            let mut contents = String::from("");
            let mut tokens: Vec<char> = vec![];
            if let Ok(_) = token_file.read_to_string(&mut contents) {
                for ch in contents.chars().into_iter() {
                    if !tokens.contains(&ch) {
                        tokens.push(ch);
                    }
                }
            }
            tokens.sort();

            match File::create(&self.tokenizer_config) {
                Ok(mut tokenizer_config) => {
                    tokenizer_config.write(tokens.into_iter()
                        .collect::<String>().as_bytes())
                        .expect("alphabet should be written complete");
                    tokenizer_config.sync_data().expect("sync should complete");
                    Ok(tokenizer_config)
                },
                Err(_) => Err(Error::new(ErrorKind::Other, format!("create {} failed", self.tokenizer_config)))
            }
        } else {
            File::open(&self.tokenizer_config)
        };
        
        if let Ok(mut config_file) = config_file {
            let mut token_str = String::new();
            if let Ok(_) = config_file.read_to_string(&mut token_str) {
                // println!("token_str:{}", token_str);
                let alphabet: Vec<char> = token_str.chars().into_iter().map(|token| token).collect();
                // println!("alphabet={:?}", alphabet);
                return Some(SimpleTokenizer::new(alphabet));
            }
        }
        None
    }
}

#[derive(new)]
pub struct SimpleTokenizer {
    alphabet: Vec<char>,
}

impl SimpleTokenizer {
    fn stoi(&self, ch: &char) -> usize {
        let mut idx = 0;
        for item in &self.alphabet {
            if item.eq(ch) {
                return idx;
            }
            idx += 1;
        }
        idx
    }

    fn itos(&self, idx: usize) -> char {
        self.alphabet.get(idx).unwrap().clone()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, value: &str) -> Vec<usize> {
        String::from(value)
            .chars()
            .into_iter()
            .map(|ch| self.stoi(&ch))
            .collect::<Vec<usize>>()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.into_iter().map(|idx| self.itos(*idx)).collect::<String>()
    }
    
    fn vocab_size(&self) -> usize {
        self.alphabet.len()
    }
}