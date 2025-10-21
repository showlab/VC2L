# VC2L:Exploring a Unified Vision-Centric Contrastive Alternatives on Multi-Modal Web Documents

This repository contains training and evaluation scripts for the VC2L framework, supporting three tasks: AnyCIR, SeqCIR, and CSR. The implementation builds upon the [OpenCLIP](https://github.com/mlfoundations/open_clip) project.

![method](https://github.com/user-attachments/assets/b2f78c90-3506-481b-91b6-cd5d743eace8)

---
## Repository Structure

```
.
├── README.md                     # Project overview and instructions
├── gen_seek.py                   # Script to generate seek maps from JSONL
├── ppt_val.txt                   # Validation split for the CSR task
├── test_AnyCIR.py                # Evaluation script for AnyCIR
├── test_CSR.py                   # Evaluation script for CSR
├── test_SeqCIR.py                # Evaluation script for SeqCIR
├── unifont-9.0.06.hex            # Font file (text rendering support)
├── unifont_upper-9.0.06.hex      # Font file (text rendering support)
├── open_clip/                    # Code adapted from OpenCLIP
└── training/                     # Training code
```

---
## Models

MMC4-core 20 epoches pretrained checkpoint can be found [here](https://drive.google.com/file/d/1iRjtzaZ4kNAdf6lDyx5_cs12sb5Tqmst/view?usp=drive_link).

---
## Datasets

### 1. V2CL Training (MMC4)

* Download MMC4 from Hugging Face:
  [https://huggingface.co/datasets/jmhessel/mmc4-core-ff](https://huggingface.co/datasets/jmhessel/mmc4-core-ff)

* Convert the annotation file to JSONL format. Example entry:

```json
{
  "text": [
    "26/02/2015… Canon PowerShot ELPH 160 PDF User Manual / Owner’s Manual / User Guide offers information and instructions how to operate the PowerShot ELPH 160..."
  ],
  "img": [
    [
      {
        "image_name": "15edefb1780c.jpg",
        "matched_text_index": 3,
        "matched_sim": 0.259
      },
      {
        "image_name": "8842c801adbe.jpg",
        "matched_text_index": 6,
        "matched_sim": 0.304
      }
    ],
    []
  ]
}
```

* Generate the seek map:

```bash
python3 gen_seek.py path_to_jsonl_file
```

---

### 2. AnyCIR and SeqCIR Evaluation (OBELICS)

* Download OBELICS from Hugging Face:
  [https://huggingface.co/datasets/HuggingFaceM4/OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)

* Convert the annotation file to JSONL format. Example entry:

```json
{
  "text": ["Jean-Paul Sartre and Simone de Beauvoir at the Balzac Memorial...", "..."],
  "img": [
    ["7b0cc10f1183..."], 
    ["b991ec778d5e..."], 
    [], 
    []
  ],
  "num_images": 3
}
```

* Generate the seek map:

```bash
python3 gen_seek.py path_to_jsonl_file
```

---

### 3. CSR Evaluation

* Download the Slide1M dataset from Stanford:
  [https://exhibits.stanford.edu/data/catalog/mv327tb8364](https://exhibits.stanford.edu/data/catalog/mv327tb8364)

---



## Getting Started

### Prerequisites

Install the required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch wandb opencv-python orjson nlpaug pathlib
```

---

## Training

To start training with multi-GPU support:

```bash
torchrun --nproc_per_node ${num_gpu} \
         --master_port 23456 \
         -m training.main_v2cl \
         --lr 1e-4 \
         --name ${exp_name} \
         --epochs 20 \
         --train-data ${dataset_name} \
         --dataset-type vc2ldataset \
         --csv_root '' \
         --save-frequency 1 \
         --batch-size 32 \
         --accum-freq 2 \
         --precision amp \
         --workers 12 \
         --model ViT-B-16-448 \
         --zeroshot-frequency 100 \
         --pretrained ${pretrained_ckpt} \
         --report-to wandb \
         --torchcompile \
         --drop_rate 0.4 \
         --aug_text 0.4 \
         --long_range 0 \
         --wandb-project-name ${wandb_name}
```

Replace placeholders such as `${num_gpu}`, `${exp_name}`, `${dataset_name}`, etc., with your actual configurations.

---

## Evaluation

### Evaluate AnyCIR

In `test_AnyCIR.py`, update the following variables:

* `OBLICES_PATH`: Path to the OBELICS dataset
* `OBLICES_SEEK_MAP_PATH`: Path to the OBELICS seek map
* `ckpt_list`: List of trained model checkpoints to evaluate

Run the script:

```bash
python test_AnyCIR.py
```

---

### Evaluate SeqCIR

In `test_SeqCIR.py`, update the following variables:

* `OBLICES_PATH`: Path to the OBELICS dataset
* `OBLICES_SEEK_MAP_PATH`: Path to the OBELICS seek map
* `ckpt_list`: List of trained model checkpoints to evaluate

Run the script:

```bash
python test_SeqCIR.py
```

---

### Evaluate CSR

In `test_CSR.py`, update the following variables:

* `DATASET_ROOT`: Path to the CSR dataset
* `ckpt_list`: List of trained model checkpoints to evaluate

Run the script:

```bash
python test_CSR.py
```