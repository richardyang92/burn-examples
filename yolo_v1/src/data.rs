use std::{fs::{read_dir, File}, io::BufReader, path::Path, usize};


use burn::{data::{dataloader::batcher::Batcher, dataset::Dataset}, tensor::{backend::Backend, Data, Int, Tensor}};
use image::{imageops::FilterType::Nearest, io::Reader as ImageReader, GenericImageView};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use xml::{common::Position, reader::XmlEvent, ParserConfig};


const WIDTH: usize = 448;
const HEIGHT: usize = 448;
const SEGMENT: usize = 7;

pub const OBJ_CLASSES: [&str; 20] = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
];

#[derive(Debug)]
pub struct VocObjBox {
    box_class: String,
    xmin: usize,
    ymin: usize,
    xmax: usize,
    ymax: usize,
}

impl VocObjBox {
    pub fn new() -> Self {
        Self {
            box_class: String::new(),
            xmin: 0,
            ymin: 0,
            xmax: 0,
            ymax: 0,
        }
    }

    fn get_label_class_idx(&self) -> usize {
        let mut class_idx = 0;

        for obj_class in OBJ_CLASSES {
            if self.box_class.as_str().eq(obj_class) {
                break;
            }
            class_idx += 1;
        }
        class_idx
    }
}

#[derive(Debug)]
pub struct VocLabel {
    filename: String,
    width: usize,
    height: usize,
    obj_boxes: Vec<VocObjBox>,
}

impl VocLabel {
    pub fn new(label_path: &Path) -> Self {
        let file = File::open(label_path).unwrap();
        let mut reader = ParserConfig::default()
            .ignore_root_level_whitespace(false)
            .create_reader(BufReader::new(file));

        let mut filename = String::new();
        let mut width = 0;
        let mut height = 0;
        let mut obj_boxes = Vec::<VocObjBox>::new();

        let mut element_name = String::new();

        loop {
            match reader.next() {
                Ok(e) => {
                    // print!("{}\t", reader.position());
    
                    match e {
                        XmlEvent::StartDocument { version, encoding, .. } => {
                            println!("StartDocument({version}, {encoding})");
                        },
                        XmlEvent::EndDocument => {
                            println!("EndDocument");
                            break;
                        }
                        XmlEvent::StartElement { name, .. } => {
                            // println!("StartElement({name})");
                            let cur_element_name = name.local_name;
                            if !element_name.eq(&cur_element_name) {
                                element_name = cur_element_name;
                            }
                            if element_name.as_str().eq("object") {
                                obj_boxes.push(VocObjBox::new());
                            }
                        }
                        XmlEvent::EndElement { .. } => {
                            // println!("EndElement({name})");
                            element_name = String::new();
                        },
                        XmlEvent::Characters(data) => {
                            // println!(r#"Characters("{}")"#, data.escape_debug());
                            let element_name_str = element_name.as_str();
                            if element_name_str.eq("filename") {
                                filename = data;
                            } else if element_name_str.eq("width") {
                                width = data.parse().unwrap_or(0);
                            } else if element_name_str.eq("height") {
                                height = data.parse().unwrap_or(0);
                            } else if element_name_str.eq("name") {
                                if let Some(obj_box) = obj_boxes.last_mut() {
                                    obj_box.box_class = data;
                                }
                            } else if element_name_str.eq("xmin") {
                                if let Some(obj_box) = obj_boxes.last_mut() {
                                    obj_box.xmin = data.parse().unwrap_or(0);
                                }
                            } else if element_name_str.eq("ymin") {
                                if let Some(obj_box) = obj_boxes.last_mut() {
                                    obj_box.ymin = data.parse().unwrap_or(0);
                                }
                            } else if element_name_str.eq("xmax") {
                                if let Some(obj_box) = obj_boxes.last_mut() {
                                    obj_box.xmax = data.parse().unwrap_or(0);
                                }
                            } else if element_name_str.eq("ymax") {
                                if let Some(obj_box) = obj_boxes.last_mut() {
                                    obj_box.ymax = data.parse().unwrap_or(0);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Err(e) => {
                    eprintln!("Error at {}: {e}", reader.position());
                    break;
                },
            }
        }
        // println!("obj_boxes: {:?}", obj_boxes);

        if width != WIDTH || height != HEIGHT {
            let (ratio_x, ratio_y) = (width as f32 / WIDTH as f32, height as f32 / HEIGHT as f32);
            for obj_box in &mut obj_boxes {
                obj_box.xmin = (obj_box.xmin as f32 * ratio_x) as usize;
                obj_box.ymin = (obj_box.ymin as f32 * ratio_y) as usize;
                obj_box.xmax = (obj_box.xmax as f32 * ratio_x) as usize;
                obj_box.ymax = (obj_box.ymax as f32 * ratio_y) as usize;
            }
        }
        Self {
            filename,
            width: WIDTH,
            height: HEIGHT,
            obj_boxes,
        }
    }
}

pub trait ItemLoader {
    fn load_image(&self) -> Option<[[[u8; WIDTH]; HEIGHT]; 3]>;
    fn load_lable(&self) -> Option<[[[f32; 30]; SEGMENT]; SEGMENT]>;
}

pub struct VocItem {
    image: [[[u8; WIDTH]; HEIGHT]; 3],
    label: [[[f32; 30]; SEGMENT]; SEGMENT],
}

#[derive(Debug, Clone)]
pub struct VocItemLoader {
    pub image_path: String,
    pub label_path: String,
}

impl ItemLoader for VocItemLoader {
    fn load_image(&self) -> Option<[[[u8; WIDTH]; HEIGHT]; 3]> {
        let image_path = &self.image_path;
        let mut jpg = ImageReader::open(image_path)
            .expect(format!("load image {} failed", image_path).as_str())
            .decode().expect(format!("decode image {} failed", image_path).as_str());

        // resize image to 448x448
        if jpg.width() != WIDTH as u32 || jpg.height() != HEIGHT as u32 {
            jpg = jpg.resize_exact(WIDTH as u32, HEIGHT as u32, Nearest);
        }
        let image_buffer = match jpg {
            image::DynamicImage::ImageRgb8(rgb) => rgb,
            _ => jpg.to_rgb8(),
        };

        let mut rgb = [[[0; WIDTH]; HEIGHT]; 3];
 
        for (x, y, pixel) in image_buffer.enumerate_pixels() {
            rgb[0][x as usize][y as usize] = pixel.0[0];
            rgb[1][x as usize][y as usize] = pixel.0[1];
            rgb[2][x as usize][y as usize] = pixel.0[2];
        }
        Some(rgb)
    }

    fn load_lable(&self) -> Option<[[[f32; 30]; SEGMENT]; SEGMENT]> {
        let label_path = &self.label_path;
        let voc_label = VocLabel::new(Path::new(label_path));
        println!("origin label: {:?}", voc_label);

        if voc_label.width != WIDTH || voc_label.height != HEIGHT {
            return None;
        }

        let mut label = [[[0f32; 30]; SEGMENT]; SEGMENT];

        let cell_size = WIDTH / SEGMENT;
        
        for obj_box in voc_label.obj_boxes {
            let class_idx = obj_box.get_label_class_idx();

            let (box_w, box_h) = (obj_box.xmax - obj_box.xmin, obj_box.ymax - obj_box.ymin);
            let (cx, cy) = (obj_box.xmin + box_w / 2, obj_box.ymin + box_h / 2);
            let (i, j) = (cx / cell_size, cy / cell_size);
            let (delta_x, delta_y) = (((cx - cell_size * i) as f32) / cell_size as f32,
                ((cy - cell_size * j) as f32) / cell_size as f32);

            label[i][j][0] = delta_x;
            label[i][j][1] = delta_y;
            label[i][j][2] = box_w as f32/ WIDTH as f32;
            label[i][j][3] = box_h as f32 / WIDTH as f32;
            label[i][j][4] = 1f32;
            label[i][j][5] = delta_x;
            label[i][j][6] = delta_y;
            label[i][j][7] = box_w as f32/ WIDTH as f32;
            label[i][j][8] = box_h as f32 / WIDTH as f32;
            label[i][j][9] = 1f32;
            label[i][j][class_idx + 9] = 1f32;
        }
        Some(label)
    }
}

pub struct YoloV1Batcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> YoloV1Batcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

pub struct YoloV1Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Batcher<VocItemLoader, YoloV1Batch<B>> for YoloV1Batcher<B> {
    fn batch(&self, items: Vec<VocItemLoader>) -> YoloV1Batch<B> {
        let images = items
            .iter()
            .filter_map(|item_path| item_path.load_image())
            .map(|image| Data::<u8, 3>::from(image))
            .map(|data| Tensor::<B, 3>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 3, HEIGHT, WIDTH]))
            .collect();

        let targets = items
            .iter()
            .filter_map(|label_path| label_path.load_lable())
            .map(|label| Data::<f32, 3>::from(label))
            .map(|data| Tensor::<B, 3>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 30, SEGMENT, SEGMENT]))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        YoloV1Batch { images, targets }
    }
}

#[derive(Debug)]
pub struct VocDataSet {
    voc_items: Vec<VocItemLoader>
}

impl Dataset<VocItem> for VocDataSet {
    fn get(&self, index: usize) -> Option<VocItem> {
        let item_path = &self.voc_items[index];
        let image = item_path.load_image();
        let label = item_path.load_lable();

        if let Some(image) = image {
            if let Some(label) = label {
                return Some(VocItem { image, label });
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.voc_items.len()
    }
}

impl VocDataSet {
    pub fn new(voc_path: &str) -> Self {
        let voc_dir = Path::new(voc_path);
        let image_dir = voc_dir.join("JPEGImages");
        let label_dir = voc_dir.join("Annotations");

        let mut image_paths = Self::load_file_names(&image_dir, "jpg");
        let mut label_paths = Self::load_file_names(&label_dir, "xml");

        image_paths.sort();
        label_paths.sort();

        let voc_items = image_paths.into_iter()
            .zip(label_paths)
            .map(|(image_path, label_path)|
                VocItemLoader {
                    image_path,
                    label_path
                })
            .collect();

        Self {
            voc_items
        }
    }

    pub fn shuffle_with_seed(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        self.voc_items.shuffle(&mut rng);
    }

    pub fn split_by_ratio(&self, ratio: usize) -> (Self, Self) {
        let total_len = self.voc_items.len();
        let split_index = (ratio * total_len / 10).min(total_len - 1);
        let (first_slice, second_silce) = self.voc_items.split_at(split_index);
        (Self { voc_items: first_slice.to_vec() }, Self { voc_items: second_silce.to_vec() })
    }

    fn load_file_names(image_dir: &Path, _suffix: &str) -> Vec<String> {
        read_dir(image_dir)
        .expect(format!("open {:?} failed", image_dir).as_str())
        .map(|entry| entry.unwrap().path())
        .filter_map(|path|
            match path.extension().and_then(|ext| ext.to_str()) {
                Some(_suffix) => Some(String::from(path.to_str().unwrap())),
                _ => None,
            })
        .collect()
    }
}