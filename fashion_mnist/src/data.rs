use std::{fs::File, io::{Read, Seek, SeekFrom}, path::Path};

use burn::{data::{dataloader::batcher::Batcher, dataset::{transform::{Mapper, MapperDataset}, vision::MNISTItem, Dataset, InMemDataset}}, serde::Deserialize, tensor::{backend::Backend, Data, ElementConversion, Int, Tensor}};

pub struct FashionMNISTBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> FashionMNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct FashionMNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MNISTItem, FashionMNISTBatch<B>> for FashionMNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> FashionMNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()]), &self.device))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        FashionMNISTBatch { images, targets }
    }
}

const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(Deserialize, Debug, Clone)]
struct MNISTItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}

struct BytesToImage;

impl Mapper<MNISTItemRaw, MNISTItem> for BytesToImage {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
    fn map(&self, item: &MNISTItemRaw) -> MNISTItem {
        // Ensure the image dimensions are correct.
        debug_assert_eq!(item.image_bytes.len(), WIDTH * HEIGHT);

        // Convert the image to a 2D array of floats.
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in item.image_bytes.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            image_array[y][x] = *pixel as f32;
        }

        MNISTItem {
            image: image_array,
            label: item.label,
        }
    }
}

type MappedDataset = MapperDataset<InMemDataset<MNISTItemRaw>, BytesToImage, MNISTItemRaw>;

pub struct FashionMNISTDataset {
    dataset: MappedDataset,
}

impl Dataset<MNISTItem> for FashionMNISTDataset {
    fn get(&self, index: usize) -> Option<MNISTItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl FashionMNISTDataset {
    /// Creates a new train dataset.
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Creates a new test dataset.
    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        // Download dataset
        let root = Path::new("./fashion_mnist/data");

        // MNIST is tiny so we can load it in-memory
        // Train images (u8): 28 * 28 * 60000 = 47.04Mb
        // Test images (u8): 28 * 28 * 10000 = 7.84Mb
        let images = FashionMNISTDataset::read_images(&root, split);
        let labels = FashionMNISTDataset::read_labels(&root, split);

        // Collect as vector of MNISTItemRaw
        let items: Vec<_> = images
            .into_iter()
            .zip(labels)
            .map(|(image_bytes, label)| MNISTItemRaw { image_bytes, label })
            .collect();

        let dataset = InMemDataset::new(items);
        let dataset = MapperDataset::new(dataset, BytesToImage);

        Self { dataset }
    }

    /// Read images at the provided path for the specified split.
    /// Each image is a vector of bytes.
    fn read_images<P: AsRef<Path>>(root: &P, split: &str) -> Vec<Vec<u8>> {
        let file_name = if split == "train" {
            TRAIN_IMAGES
        } else {
            TEST_IMAGES
        };
        let file_name = root.as_ref().join(file_name);
        // Read number of images from 16-byte header metadata
        let mut f = File::open(file_name).unwrap();
        let mut buf = [0u8; 4];
        let _ = f.seek(SeekFrom::Start(4)).unwrap();
        f.read_exact(&mut buf)
            .expect("Should be able to read image file header");
        let size = u32::from_be_bytes(buf);

        let mut buf_images: Vec<u8> = vec![0u8; WIDTH * HEIGHT * (size as usize)];
        let _ = f.seek(SeekFrom::Start(16)).unwrap();
        f.read_exact(&mut buf_images)
            .expect("Should be able to read image file header");

        buf_images
            .chunks(WIDTH * HEIGHT)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Read labels at the provided path for the specified split.
    fn read_labels<P: AsRef<Path>>(root: &P, split: &str) -> Vec<u8> {
        let file_name = if split == "train" {
            TRAIN_LABELS
        } else {
            TEST_LABELS
        };
        let file_name = root.as_ref().join(file_name);

        // Read number of labels from 8-byte header metadata
        let mut f = File::open(file_name).unwrap();
        let mut buf = [0u8; 4];
        let _ = f.seek(SeekFrom::Start(4)).unwrap();
        f.read_exact(&mut buf)
            .expect("Should be able to read label file header");
        let size = u32::from_be_bytes(buf);

        let mut buf_labels: Vec<u8> = vec![0u8; size as usize];
        let _ = f.seek(SeekFrom::Start(8)).unwrap();
        f.read_exact(&mut buf_labels)
            .expect("Should be able to read labels from file");

        buf_labels
    }
}