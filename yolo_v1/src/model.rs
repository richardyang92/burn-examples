use std::{cmp::{max_by, min_by}, fmt::Display};

use burn::{config::Config, module::Module, nn::{conv::{Conv2d, Conv2dConfig}, pool::{MaxPool2d, MaxPool2dConfig}, Linear, LinearConfig, PaddingConfig2d}, tensor::{backend::{AutodiffBackend, Backend}, Float, Tensor}, train::{metric::{Adaptor, LossInput}, RegressionOutput, TrainOutput, TrainStep, ValidStep}};

use crate::data::YoloV1Batch;

use derive_new::new;
use itertools::iproduct;

#[derive(Debug, Clone, Copy, new)]
pub struct BBox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    prob: f32,
}

impl From<&[f32]> for BBox {
    fn from(value: &[f32]) -> Self {
        BBox {
            xmin: value[0],
            ymin: value[1],
            xmax: value[2],
            ymax: value[3],
            prob: value[4],
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
            - max_by(box1.ymin, box2.ymin, comparator), comparator);
        let w = max_by(0f32, min_by(box1.xmax, box2.xmax, comparator)
            - max_by(box1.xmin, box2.xmin, comparator), comparator);
        let inter = h * w;
        let area_box1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin);
        let area_box2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin);
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
        
        (self.has_object(probs), (BBox::from(box_1), BBox::from(box_2)), prob_arr)
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
                if target_has_obj {
                    let iou_1 = self.compute_iou::<B>(predict_bbox_1, target_bbox);
                    let iou_2 = self.compute_iou::<B>(predict_bbox_2, target_bbox);
                    let choosen_bbox = if iou_1 > iou_2 {
                        loss += (predict_bbox_1.prob - target_bbox.prob).powi(2);
                        loss += self.l_noobj * (predict_bbox_2.prob - target_bbox.prob).powi(2);
                        predict_bbox_1
                    } else {
                        loss += self.l_noobj * (predict_bbox_1.prob - target_bbox.prob).powi(2);
                        loss += (predict_bbox_2.prob - target_bbox.prob).powi(2);
                        predict_bbox_2
                    };
                    loss += self.l_coord * ((choosen_bbox.xmin - target_bbox.xmin).powi(2)
                        + (choosen_bbox.xmax - target_bbox.xmax).powi(2)
                        + (choosen_bbox.ymin - target_bbox.ymin).powi(2)
                        + (choosen_bbox.ymax - target_bbox.ymax).powi(2));
                    let (wc_sqrt, wt_sqrt) = ((choosen_bbox.xmax - choosen_bbox.xmin).abs().sqrt(), (target_bbox.xmax - target_bbox.xmax).abs().sqrt());
                    let (hc_sqrt, ht_sqrt) = ((choosen_bbox.ymax - choosen_bbox.ymin).abs().sqrt(), (target_bbox.ymax - target_bbox.ymin).abs().sqrt());
                    loss += self.l_coord * ((wc_sqrt - wt_sqrt).powi(2) + (hc_sqrt - ht_sqrt).powi(2));
                    for i in 0..20 {
                        loss += (predict_probs[i] - target_probs[i]).powi(2);
                    }
                } else {
                    loss += self.l_noobj * (predict_bbox_1.prob - target_bbox.prob).powi(2);
                    loss += self.l_noobj * (predict_bbox_2.prob - target_bbox.prob).powi(2);
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
        let x = self.fc2.forward(x);
        x.reshape([batch_size, 30, 7, 7])
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