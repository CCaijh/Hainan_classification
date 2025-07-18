# Hainan_classification

This repository contains everything you need to **train, evaluate, and analyse** a Self-Attention ResNet-50 (SA-ResNet50) model for image–classification tasks—originally designed for frequency-spectrogram data from **2016**.  
It bundles reproducible configuration files, data-handling utilities, and helper scripts to visualise both learning-rate schedules and loss curves.

---

## 1. Quick Start

```bash
# 1️⃣  Create a Python environment (Python ≥ 3.9 recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2️⃣  Install dependencies
pip install -r requirements.txt    # supply your own requirements if not yet created

# 3️⃣  Split your dataset (80 % train / 20 % val by default)
python split_data.py --data_root ./dataset

# 4️⃣  Train SA-ResNet50
python train.py --config configs_0_5t.py

# 5️⃣  Plot learning-rate & loss history
python loss_lr_plt.py --log_dir ./logs

# 6️⃣  Run inference on new samples
python predict.py --model_path ./model/best.pth --input ./example.jpg
2. Repository Contents
File / Folder	Purpose
model/	Stores trained checkpoints (*.pth) and exported ONNX/TorchScript models.
2016_sa_resnet50Net.txt	Predicted class labels produced by SA-ResNet50 on the 2016 evaluation set.
2016_Ground_Truth.txt	Expert-annotated ground-truth labels for the same 2016 set—used for benchmarking.
configs_0_5t.py	Canonical hyper-parameter file (batch-size, LR schedule, optimiser, data-paths, etc.).
loss_lr_plt.py	Utility to parse training logs and draw loss & learning-rate curves.
part_random_noise_newest_uniform.py	Optional data-augmentation script that injects uniform random noise.
predict.py	Stand-alone inference script—loads a checkpoint and outputs predicted labels/scores.
split_data.py	Splits a folder of images into train and val sub-folders in an 8:2 ratio.
train.py	The main training loop (supports resuming, mixed-precision, and configurable metrics).
train_order.txt	Fixed batch ordering used in the original experiments (helps exact reproducibility).

3. Dataset Preparation
Put all raw images in ./dataset/original/.

Run:

bash
python split_data.py --data_root ./dataset/original --train_ratio 0.8
This generates:

kotlin
dataset/
├── train/
└── val/
Update any custom paths inside configs_0_5t.py if your folder layout differs.

4. Training
bash
复制
编辑
python train.py --config configs_0_5t.py \
                --epochs 200 \
                --seed 42
Mixed precision is enabled by default (set fp16=False in the config to disable).

Checkpoints are saved to ./model/ every save_interval epochs and whenever validation accuracy improves.

5. Evaluation & Re-Scoring Existing Results
bash
复制
编辑
# Calculate top-1 & top-k accuracy on the 2016 set
python evaluate_txt.py \
       --pred 2016_sa_resnet50Net.txt \
       --gt   2016_Ground_Truth.txt
6. Visualising the Training Process
bash
python loss_lr_plt.py --log_dir ./logs --out_fig ./plots/training_curves.png
Generates a side-by-side plot of:

Training & validation loss

Learning-rate schedule (supports cosine, step, and custom warm-up)

7. Adding Noise-Augmented Samples (Optional)
bash
python part_random_noise_newest_uniform.py \
       --src_dir ./dataset/train \
       --dst_dir ./dataset/train_noise \
       --sigma 0.05
This produces noise-augmented images in a parallel directory that can be referenced in your custom config to enrich the training set.

8. Reproducing the Published 2016 Numbers
Download the released checkpoint into ./model/2016_sa_resnet50.pth.

Run:

bash
python predict.py \
       --model_path ./model/2016_sa_resnet50.pth \
       --data_root ./dataset/2016_eval \
       --out_file 2016_sa_resnet50Net.txt
Compare with ground truth using evaluate_txt.py (see Section 5).

9. Citations
If you use this code in your research, please cite:

10. License
Distributed under the MIT License.
See LICENSE for more information.

11. Acknowledgements
This project was initially developed at Communication University of China.
We thank the annotators who produced 2016_Ground_Truth.txt and the open-source community for their invaluable contributions.

