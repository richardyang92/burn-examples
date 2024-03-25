# burn-examples
some neural networks implemented by burn framework:

## 1. bigram

   **runing cmd**: `cargo run --package bigram`
   
   **training result**: accuracy is about 35% for 10 epochs training, quite normal for this simplest language model.
   
## 2. fashion mnist
   
   **runing cmd**: `cargo run --package fashion_mnist`
   
   **training result**: the model works well.
   
## 3. yolo v1
   
   **runing cmd**: `cargo run --package yolo_v1`
   
   **training result**: Training process can't be completed as the loss stops decreasing at around 14, leading to its termination by the burn framework. I'm still investigating the reasons behind this.
