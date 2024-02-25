use burn::{config::Config, module::Module, nn::{conv::{Conv2d, Conv2dConfig}, pool::{MaxPool2d, MaxPool2dConfig}, Linear, LinearConfig, PaddingConfig2d}, tensor::{backend::{AutodiffBackend, Backend}, Float, Tensor}, train::{metric::{Adaptor, LossInput}, TrainOutput, TrainStep, ValidStep}};

use crate::data::YoloV1Batch;

use derive_new::new;

#[derive(Module, Debug, Clone, new)]
pub struct YoloV1Loss {
    segments: usize,
    boxes: usize,
    l_coord: usize,
    l_noobj: f32,
}

impl YoloV1Loss {

    fn compute_iou<B: Backend>(&self, boxes_1: Vec<[f32; 4]>, boxes_2: Vec<[f32; 4]>) -> Tensor<B, 2> {
        todo!()
    }

    pub fn forward<B: Backend>(&self, image: Tensor<B, 4>, target: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 1, Float> {
        todo!()
    }
}

#[derive(Config)]
pub struct YoloV1LossConfig {
    segments: usize,
    boxes: usize,
    l_coord: usize,
    l_noobj: f32,
}

impl YoloV1LossConfig {
    pub fn init<B: Backend>(&self) -> YoloV1Loss {
        YoloV1Loss {
            segments: self.segments,
            boxes: self.boxes,
            l_coord: self.l_coord,
            l_noobj: self.l_noobj,
        }
    }
}

#[derive(new)]
pub struct YoloV1RegressionOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 4>,

    /// The targets.
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Adaptor<LossInput<B>> for YoloV1RegressionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Module, Debug)]
pub struct YoloV1<B: Backend> {
    conv1: Conv2d<B>,
    pool1: MaxPool2d,
    conv2: Conv2d<B>,
    pool2: MaxPool2d,
    conv3_1: Conv2d<B>,
    conv3_2: Conv2d<B>,
    conv3_3: Conv2d<B>,
    conv3_4: Conv2d<B>,
    pool3: MaxPool2d,
    conv4_1: Conv2d<B>,
    conv4_2: Conv2d<B>,
    conv4_3: Conv2d<B>,
    conv4_4: Conv2d<B>,
    conv4_5: Conv2d<B>,
    conv4_6: Conv2d<B>,
    conv4_7: Conv2d<B>,
    conv4_8: Conv2d<B>,
    conv4_9: Conv2d<B>,
    conv4_10: Conv2d<B>,
    pool4: MaxPool2d,
    conv5_1: Conv2d<B>,
    conv5_2: Conv2d<B>,
    conv5_3: Conv2d<B>,
    conv5_4: Conv2d<B>,
    conv5_5: Conv2d<B>,
    pool5: MaxPool2d,
    conv6_1: Conv2d<B>,
    conv6_2: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> YoloV1<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(input);
        let x = self.pool1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.pool2.forward(x);
        let x = self.conv3_1.forward(x);
        let x = self.conv3_2.forward(x);
        let x = self.conv3_3.forward(x);
        let x = self.conv3_4.forward(x);
        let x= self.pool3.forward(x);
        let x = self.conv4_1.forward(x);
        let x = self.conv4_2.forward(x);
        let x = self.conv4_3.forward(x);
        let x = self.conv4_4.forward(x);
        let x = self.conv4_5.forward(x);
        let x = self.conv4_6.forward(x);
        let x = self.conv4_7.forward(x);
        let x = self.conv4_8.forward(x);
        let x = self.conv4_9.forward(x);
        let x = self.conv4_10.forward(x);
        let x = self.pool4.forward(x);
        let x = self.conv5_1.forward(x);
        let x = self.conv5_2.forward(x);
        let x = self.conv5_3.forward(x);
        let x = self.conv5_4.forward(x);
        let x = self.pool5.forward(x);
        let x = self.conv6_1.forward(x);
        let x = self.conv6_2.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);
        let x = self.fc1.forward(x);
        self.fc2.forward(x).reshape([batch_size, 30, 7, 7])
    }

    pub fn forward_regression(&self, images: Tensor<B, 4>, targets: Tensor<B, 4>) -> YoloV1RegressionOutput<B> {
        let output = self.forward(images);
        let loss = YoloV1LossConfig::new(7, 2, 5, 0.5).init::<B>().forward(output.clone(), targets.clone(), &output.device());
        YoloV1RegressionOutput {
            loss, output, targets
        }
    }

    
}

#[derive(Config)]
pub struct YoloV1Config {
    segments_number: usize,
    boxes_number: usize,
}

impl YoloV1Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YoloV1<B> {
        let conv1 = Conv2dConfig::new([3, 64], [self.segments_number, self.segments_number])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_stride([2, 2]).init(device);
        let pool1 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv2 = Conv2dConfig::new([64, 192], [3, 3]).init(device);
        let pool2 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv3_1 = Conv2dConfig::new([192, 128], [1, 1]).init(device);
        let conv3_2 = Conv2dConfig::new([128, 256], [3, 3]).init(device);
        let conv3_3 = Conv2dConfig::new([256, 256], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).init(device);
        let conv3_4 = Conv2dConfig::new([256, 512], [3, 3]).init(device);
        let pool3 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv4_1 = Conv2dConfig::new([512, 256], [1, 1]).init(device);
        let conv4_2 = Conv2dConfig::new([256, 512], [3, 3]).init(device);
        let conv4_3 = Conv2dConfig::new([512, 256], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).init(device);
        let conv4_4 = Conv2dConfig::new([256, 512], [3, 3]).init(device);
        let conv4_5 = Conv2dConfig::new([512, 256], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).init(device);
        let conv4_6 = Conv2dConfig::new([256, 512], [3, 3]).init(device);
        let conv4_7 = Conv2dConfig::new([512, 256], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).init(device);
        let conv4_8 = Conv2dConfig::new([256, 512], [3, 3]).init(device);
        let conv4_9 = Conv2dConfig::new([512, 512], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).init(device);
        let conv4_10 = Conv2dConfig::new([512, 1024], [3, 3]).init(device);
        let pool4 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        
        let conv5_1 = Conv2dConfig::new([1024, 512], [1, 1]).init(device);
        let conv5_2 = Conv2dConfig::new([512, 1024], [3, 3]).init(device);
        let conv5_3 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).init(device);
        let conv5_4 = Conv2dConfig::new([512, 1024], [3, 3]).init(device);
        let conv5_5 = Conv2dConfig::new([1024, 1024], [3, 3]).init(device);
        let pool5 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv6_1 = Conv2dConfig::new([1024, 1024], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).init(device);
        let conv6_2 = Conv2dConfig::new([1024, 1024], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).init(device);
        let fc1 = LinearConfig::new(1024 * self.segments_number * self.segments_number, 4096).init(device);
        let fc2 = LinearConfig::new(4096, 30 * self.segments_number * self.segments_number).init(device);

        YoloV1 {
            conv1,
            pool1,
            conv2,
            pool2,
            conv3_1,
            conv3_2,
            conv3_3,
            conv3_4,
            pool3,
            conv4_1,
            conv4_2,
            conv4_3,
            conv4_4,
            conv4_5,
            conv4_6,
            conv4_7,
            conv4_8,
            conv4_9,
            conv4_10,
            pool4,
            conv5_1,
            conv5_2,
            conv5_3,
            conv5_4,
            conv5_5,
            pool5,
            conv6_1,
            conv6_2,
            fc1,
            fc2,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<YoloV1Batch<B>, YoloV1RegressionOutput<B>> for YoloV1<B> {
    fn step(&self, batch: YoloV1Batch<B>) -> burn::train::TrainOutput<YoloV1RegressionOutput<B>> {
        let item = self.forward_regression(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<YoloV1Batch<B>, YoloV1RegressionOutput<B>> for YoloV1<B> {
    fn step(&self, batch: YoloV1Batch<B>) -> YoloV1RegressionOutput<B> {
        self.forward_regression(batch.images, batch.targets)
    }
}