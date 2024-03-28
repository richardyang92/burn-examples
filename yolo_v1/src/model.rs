use std::{cmp::{max_by, min_by}, fmt::Display};

use burn::{config::Config, module::Module, nn::{conv::{Conv2d, Conv2dConfig}, pool::{MaxPool2d, MaxPool2dConfig}, BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d}, tensor::{backend::{AutodiffBackend, Backend}, ElementConversion, Float, Tensor}, train::{metric::{Adaptor, LossInput}, TrainOutput, TrainStep, ValidStep}};

use crate::data::YoloV1Batch;

use derive_new::new;
use itertools::iproduct;

const WIDTH: usize = 448;
const HEIGHT: usize = 448;
const SEGMENT: usize = 7;

#[derive(Debug, Clone, Copy, new)]
pub struct BBox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    confident: f32,
    box_origin: [f32; 4],
}

impl From<(&[f32], usize, usize)> for BBox {
    fn from(value: (&[f32], usize, usize)) -> Self {
        let (box_val, i, j) = (value.0, value.1, value.2);
        let box_w = box_val[2] * WIDTH as f32;
        let box_h = box_val[3] * HEIGHT as f32;
        let cell_size_w = (WIDTH / SEGMENT) as f32;
        let cell_size_h = (HEIGHT / SEGMENT) as f32;
        let xmin = box_val[0] * cell_size_w + cell_size_w * i as f32 - box_w / 2f32;
        let ymin = box_val[1] * cell_size_h + cell_size_h * j as f32 - box_h / 2f32;

        let mut box_origin = [
            box_val[0],
            box_val[1],
            box_val[2],
            box_val[3]
        ];

        if box_origin[2] < 0f32 {
            box_origin[2] = 0f32;
        }

        if box_origin[3] < 0f32 {
            box_origin[3] = 0f32;
        }

        BBox {
            xmin,
            ymin,
            xmax: xmin + box_w,
            ymax: ymin + box_h,
            confident: box_val[4],
            box_origin
        }
    }
}

#[derive(Module, Debug, Clone, new)]
pub struct YoloV1Loss {
    l_coord: f32,
    l_noobj: f32,
}

impl YoloV1Loss {
    fn has_object(&self, probs: &[f32]) -> bool {
        let mut has_obj = false;

        for i in 0..20 {
            if probs[i] != 0f32 {
                has_obj = true;
                break;
            }
        }
        has_obj
    }

    fn compute_iou<B: Backend>(&self, box1: BBox, box2: BBox) -> f32 {
        let comparator = |f1: &f32, f2: &f32| { f1.partial_cmp(f2).unwrap() };
        let h = max_by(0f32, min_by(box1.ymax, box2.ymax, comparator)
            - max_by(box1.ymin, box2.ymin, comparator) + 1f32, comparator);
        let w = max_by(0f32, min_by(box1.xmax, box2.xmax, comparator)
            - max_by(box1.xmin, box2.xmin, comparator) + 1f32, comparator);
        let inter = h * w;
        let area_box1 = (box1.xmax - box1.xmin).abs() * (box1.ymax - box1.ymin).abs();
        let area_box2 = (box2.xmax - box2.xmin).abs() * (box2.ymax - box2.ymin).abs();
        let union = area_box1 + area_box2 - inter;
        inter / union
    }

    fn create_bboxes<B: Backend>(&self, predict: &Tensor<B, 3>, i: usize, j: usize) -> (bool, (BBox, BBox), [f32; 20]) {
        let label_slice: Tensor<B, 1> = predict.clone().slice([0..30, i..i+1, j..j+1]).flatten::<1>(0, 2);
        let label_vec = label_slice.to_data().convert::<f32>().value;
        let box_1 = &label_vec[0..5];
        let box_2 = &label_vec[5..10];
        let probs = &label_vec[10..30];

        let mut prob_arr = [0f32; 20];
        prob_arr.copy_from_slice(probs);
        
        (self.has_object(probs), (BBox::from((box_1, i, j)), BBox::from((box_2, i, j))), prob_arr)
    }

    pub fn forward<B: Backend>(&self, predicts: Tensor<B, 4>, targets: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 1, Float> {
        let predicts = predicts.detach();
        let [_, _, segment_h, segment_w] = predicts.dims();
        let predict_vec: Vec<Tensor<B, 3>> = predicts.iter_dim(0).map(|predict| predict.squeeze(0)).collect();
        let target_vec: Vec<Tensor<B, 3>> = targets.iter_dim(0).map(|target| target.squeeze(0)).collect();

        let batch_items: Vec<(Tensor<B, 3>, Tensor<B, 3>)> = predict_vec.into_iter().zip(target_vec).collect();

        let mut loss_vec: Vec<f32> = vec![];

        for (predict, target) in batch_items.into_iter() {
            let mut loss = 0f32;
            for (i, j) in iproduct!(0..segment_w, 0..segment_h) {
                let (target_has_obj, (target_bbox, _), target_probs) = self.create_bboxes(&target, i, j);
                let (_, (predict_bbox_1, predict_bbox_2), predict_probs) = self.create_bboxes(&predict, i, j);

                let iou_1 = self.compute_iou::<B>(predict_bbox_1, target_bbox);
                let iou_2 = self.compute_iou::<B>(predict_bbox_2, target_bbox);

                let (resp_bbox, no_resp_bbox) = if iou_1 > iou_2 {
                    (predict_bbox_1, predict_bbox_2)
                } else {
                    (predict_bbox_2, predict_bbox_1)
                };

                if target_has_obj {
                    loss += self.l_coord * ((resp_bbox.box_origin[0] - target_bbox.box_origin[0])).powi(2);
                    loss += self.l_coord * ((resp_bbox.box_origin[1] - target_bbox.box_origin[1])).powi(2);
                    loss += self.l_coord * (resp_bbox.box_origin[2].sqrt() - target_bbox.box_origin[2].sqrt()).powi(2);
                    loss += self.l_coord * (resp_bbox.box_origin[3].sqrt() - target_bbox.box_origin[3].sqrt()).powi(2);
                    loss += (resp_bbox.confident - target_bbox.confident).powi(2);
                    loss += self.l_noobj * (no_resp_bbox.confident - 0f32).powi(2);
                } else {
                    loss += self.l_noobj * (resp_bbox.confident - 0f32).powi(2);
                    loss += self.l_noobj * (no_resp_bbox.confident - 0f32).powi(2);
                }

                for i in 0..20 {
                    if target_probs[i] > 0f32 {
                        loss += (predict_probs[i] - target_probs[i]).powi(2);
                    }
                }
            }
            loss_vec.push(loss);
        }

        Tensor::<B, 1, Float>::from_floats(loss_vec.as_slice(), device).set_require_grad(true)
    }
}

#[derive(Config)]
pub struct YoloV1LossConfig {
    #[config(default = 5.0)]
    l_coord: f32,
    #[config(default = 0.5)]
    l_noobj: f32,
}

impl YoloV1LossConfig {
    pub fn init(&self) -> YoloV1Loss {
        YoloV1Loss {
            l_coord: self.l_coord,
            l_noobj: self.l_noobj,
        }
    }
}

#[derive(Debug)]
pub struct YoloV1RegressionOutput<B: Backend> {
    /// The loss.
    loss: Tensor<B, 1, Float>,
    /// The output.
    pub output: Tensor<B, 4>,

    /// The targets.
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Display for YoloV1RegressionOutput<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let loss_val: Vec<f32> = self.loss.to_data().convert().value;
        write!(f, "{}", loss_val[0])
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for YoloV1RegressionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Module, Clone, Debug, Default, new)]
pub struct LeakyReLU { }

impl LeakyReLU {
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let input_primitive = input.clone().into_primitive();
        let negative_slope = input.mul_scalar(0.01).into_primitive();
        let mask = B::float_lower_equal_elem(input_primitive.clone(), 0.elem());
        Tensor::new(B::float_mask_where(input_primitive, mask, negative_slope))
    }
}

#[derive(Config)]
pub struct ConvBlock2dConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 0)]
    padding: usize,
}

impl ConvBlock2dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock2d<B> {
        let conv2d = Conv2dConfig::new([self.in_channels, self.out_channels], [self.kernel_size, self.kernel_size])
                .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding)).with_stride([self.stride, self.stride]).init(device);
        let batch_norm = BatchNormConfig::new(self.out_channels).init(device);
        let leaky_relu = LeakyReLU::new();
        ConvBlock2d {
            conv2d, batch_norm, leaky_relu,
        }
    }

    pub fn init_with<B: Backend>(&self, record: ConvBlock2dRecord<B>) -> ConvBlock2d<B> {
        let conv2d = Conv2dConfig::new([self.in_channels, self.out_channels], [self.kernel_size, self.kernel_size])
                .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding)).with_stride([self.stride, self.stride]).init_with(record.conv2d);
        let batch_norm = BatchNormConfig::new(self.out_channels).init_with(record.batch_norm);
        let leaky_relu = LeakyReLU::new();
        ConvBlock2d {
            conv2d, batch_norm, leaky_relu,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock2d<B: Backend> {
    conv2d: Conv2d<B>,
    batch_norm: BatchNorm<B, 2>,
    leaky_relu: LeakyReLU,
}

impl<B: Backend> ConvBlock2d<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv2d.forward(input);
        let x = self.batch_norm.forward(x);
        self.leaky_relu.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct YoloV1<B: Backend> {
    conv1: ConvBlock2d<B>,
    pool1: MaxPool2d,
    conv2: ConvBlock2d<B>,
    pool2: MaxPool2d,
    conv3_1: ConvBlock2d<B>,
    conv3_2: ConvBlock2d<B>,
    conv3_3: ConvBlock2d<B>,
    conv3_4: ConvBlock2d<B>,
    pool3: MaxPool2d,
    conv4_1: ConvBlock2d<B>,
    conv4_2: ConvBlock2d<B>,
    conv4_3: ConvBlock2d<B>,
    conv4_4: ConvBlock2d<B>,
    conv4_5: ConvBlock2d<B>,
    conv4_6: ConvBlock2d<B>,
    conv4_7: ConvBlock2d<B>,
    conv4_8: ConvBlock2d<B>,
    conv4_9: ConvBlock2d<B>,
    conv4_10: ConvBlock2d<B>,
    pool4: MaxPool2d,
    conv5_1: ConvBlock2d<B>,
    conv5_2: ConvBlock2d<B>,
    conv5_3: ConvBlock2d<B>,
    conv5_4: ConvBlock2d<B>,
    conv5_5: ConvBlock2d<B>,
    pool5: MaxPool2d,
    conv6_1: ConvBlock2d<B>,
    conv6_2: ConvBlock2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
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
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);
        x.reshape([batch_size, 30, SEGMENT, SEGMENT])
    }

    pub fn forward_regression(&self, images: Tensor<B, 4>, targets: Tensor<B, 4>) -> YoloV1RegressionOutput<B> {
        let output = self.forward(images);
        let loss = YoloV1LossConfig::new().init().forward(output.clone(), targets.clone(), &output.device());
        YoloV1RegressionOutput {
            loss, output, targets,
        }
    }

    
}

#[derive(Config)]
pub struct YoloV1Config {
    segments_number: usize,
    boxes_number: usize,
    #[config(default = 0.3)]
    dropout_prob: f64,
}

impl YoloV1Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> YoloV1<B> {
        let conv1 = ConvBlock2dConfig::new(3, 64, self.segments_number)
            .with_padding(1).with_stride(2).init(device);
        let pool1 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv2 = ConvBlock2dConfig::new(64, 192, 3).init(device);
        let pool2 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv3_1 = ConvBlock2dConfig::new(192, 128, 1).init(device);
        let conv3_2 = ConvBlock2dConfig::new(128, 256, 3).init(device);
        let conv3_3 = ConvBlock2dConfig::new(256, 256, 1).with_padding(1).init(device);
        let conv3_4 = ConvBlock2dConfig::new(256, 512, 3).init(device);
        let pool3 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv4_1 = ConvBlock2dConfig::new(512, 256, 1).init(device);
        let conv4_2 = ConvBlock2dConfig::new(256, 512, 3).init(device);
        let conv4_3 = ConvBlock2dConfig::new(512, 256, 1).with_padding(1).init(device);
        let conv4_4 = ConvBlock2dConfig::new(256, 512, 3).init(device);
        let conv4_5 = ConvBlock2dConfig::new(512, 256, 1).with_padding(1).init(device);
        let conv4_6 = ConvBlock2dConfig::new(256, 512, 3).init(device);
        let conv4_7 = ConvBlock2dConfig::new(512, 256, 1).with_padding(1).init(device);
        let conv4_8 = ConvBlock2dConfig::new(256, 512, 3).init(device);
        let conv4_9 = ConvBlock2dConfig::new(512, 512, 1).with_padding(1).init(device);
        let conv4_10 = ConvBlock2dConfig::new(512, 1024, 3).init(device);
        let pool4 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        
        let conv5_1 = ConvBlock2dConfig::new(1024, 512, 1).init(device);
        let conv5_2 = ConvBlock2dConfig::new(512, 1024, 3).init(device);
        let conv5_3 = ConvBlock2dConfig::new(1024, 512, 1).with_padding(1).init(device);
        let conv5_4 = ConvBlock2dConfig::new(512, 1024, 3).init(device);
        let conv5_5 = ConvBlock2dConfig::new(1024, 1024, 3).init(device);
        let pool5 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv6_1 = ConvBlock2dConfig::new(1024, 1024, 3).with_padding(1).init(device);
        let conv6_2 = ConvBlock2dConfig::new(1024, 1024, 3).with_padding(1).init(device);
        let fc1 = LinearConfig::new(1024 * self.segments_number * self.segments_number, 4096).init(device);
        let fc2 = LinearConfig::new(4096, 30 * self.segments_number * self.segments_number).init(device);
        let dropout = DropoutConfig::new(self.dropout_prob).init();

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
            dropout,
        }
    }

    pub fn init_with<B: Backend>(&self, record: YoloV1Record<B>) -> YoloV1<B> {
        let conv1 = ConvBlock2dConfig::new(3, 64, self.segments_number)
            .with_padding(1).with_stride(2).init_with(record.conv1);
        let pool1 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv2 = ConvBlock2dConfig::new(64, 192, 3).init_with(record.conv2);
        let pool2 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv3_1 = ConvBlock2dConfig::new(192, 128, 1).init_with(record.conv3_1);
        let conv3_2 = ConvBlock2dConfig::new(128, 256, 3).init_with(record.conv3_2);
        let conv3_3 = ConvBlock2dConfig::new(256, 256, 1).with_padding(1).init_with(record.conv3_3);
        let conv3_4 = ConvBlock2dConfig::new(256, 512, 3).init_with(record.conv3_4);
        let pool3 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv4_1 = ConvBlock2dConfig::new(512, 256, 1).init_with(record.conv4_1);
        let conv4_2 = ConvBlock2dConfig::new(256, 512, 3).init_with(record.conv4_2);
        let conv4_3 = ConvBlock2dConfig::new(512, 256, 1).init_with(record.conv4_3);
        let conv4_4 = ConvBlock2dConfig::new(256, 512, 3).init_with(record.conv4_4);
        let conv4_5 = ConvBlock2dConfig::new(512, 256, 1).init_with(record.conv4_5);
        let conv4_6 = ConvBlock2dConfig::new(256, 512, 3).init_with(record.conv4_6);
        let conv4_7 = ConvBlock2dConfig::new(512, 256, 1).init_with(record.conv4_7);
        let conv4_8 = ConvBlock2dConfig::new(256, 512, 3).init_with(record.conv4_8);
        let conv4_9 = ConvBlock2dConfig::new(512, 512, 1).init_with(record.conv4_9);
        let conv4_10 = ConvBlock2dConfig::new(512, 1024, 3).init_with(record.conv4_10);
        let pool4 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        
        let conv5_1 = ConvBlock2dConfig::new(1024, 512, 1).init_with(record.conv5_1);
        let conv5_2 = ConvBlock2dConfig::new(512, 1024, 3).init_with(record.conv5_2);
        let conv5_3 = ConvBlock2dConfig::new(1024, 512, 1).init_with(record.conv5_3);
        let conv5_4 = ConvBlock2dConfig::new(512, 1024, 3).init_with(record.conv5_4);
        let conv5_5 = ConvBlock2dConfig::new(1024, 1024, 3).init_with(record.conv5_5);
        let pool5 = MaxPool2dConfig::new([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1)).with_strides([2, 2]).init();
        let conv6_1 = ConvBlock2dConfig::new(1024, 1024, 3).with_padding(1).init_with(record.conv6_1);
        let conv6_2 = ConvBlock2dConfig::new(1024, 1024, 3).with_padding(1).init_with(record.conv6_2);
        let fc1 = LinearConfig::new(1024 * self.segments_number * self.segments_number, 4096).init_with(record.fc1);
        let fc2 = LinearConfig::new(4096, 30 * self.segments_number * self.segments_number).init_with(record.fc2);
        let dropout = DropoutConfig::new(self.dropout_prob).init();

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
            dropout,
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