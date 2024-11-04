# VrdONE: One-stage Video Visual Relation Detection
Pytorch Implementation of ACM MM 2024 paper **"VrdONE: One-stage Video Visual Relation Detection"**.

<!-- <p align="center">
<a href="https://arxiv.org/abs/2104.14222"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
<a href="https://opensource.org/licenses/MIT"><img  src="https://img.shields.io/badge/license-MIT-blue"></a>
<a href="https://dl.acm.org/doi/10.1145/3474085.3475512"><img  src="https://img.shields.io/static/v1?label=inproceedings&message=Paper&color=orange"></a>
<a href="https://paperswithcode.com/sota/image-matting-on-p3m-10k"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/privacy-preserving-portrait-matting/image-matting-on-p3m-10k"></a>
</p> -->

[[`arXiv`]](https://arxiv.org/abs/2408.09408) [[`ACM MM`]](https://dl.acm.org/doi/10.1145/3664647.3680833)

<div align="center">
  <img src="./assets/pipeline.png" width="100%" height="100%"/>
</div><br/>


## Abstract

Video Visual Relation Detection (VidVRD) focuses on understanding how entities interact over time and space in videos, a key step for gaining deeper insights into video scenes beyond basic visual tasks. Traditional methods for VidVRD, challenged by its complexity, typically split the task into two parts: one for identifying what relation categories are present and another for determining their temporal boundaries. This split overlooks the inherent connection between these elements. Addressing the need to recognize entity pairs' spatiotemporal interactions across a range of durations, we propose VrdONE, a streamlined yet efficacious one-stage model. VrdONE combines the features of subjects and objects, turning predicate detection into 1D instance segmentation on their combined representations. This setup allows for both relation category identification and binary mask generation in one go, eliminating the need for extra steps like proposal generation or post-processing. VrdONE facilitates the interaction of features across various frames, adeptly capturing both short-lived and enduring relations. Additionally, we introduce the Subject-Object Synergy (SOS) module, enhancing how subjects and objects perceive each other before combining. VrdONE achieves state-of-the-art performances on the VidOR benchmark and ImageNet-VidVRD, showcasing its superior capability in discerning relations across different temporal scales.

## Todo List
- [x] Installation
- [x] prepare VidOR dataset
- [ ] prepare ImageNet-VidVRD datset
- [X] train VrdONE on VidOR
- [X] train VrdONE-X on VidOR
- [ ] train VrdONE on ImageNet-VidVRD
- [x] evaluate VrdONE on VidOR
- [ ] evaluate VrdONE on ImageNet-VidVRD

## Installation

1. This repository needs `python=3.10.14`, `pytorch=1.12.1`, and `torchvision=0.13.1`
2. Run the following command to install the required packages.
   ```
   pip install -r requirements.txt
   ```
3. Clone Shang's evaluation helper https://github.com/xdshang/VidVRD-helper to the root path.


## Data Preparation

Install `ffmpeg` using `sudo apt-get install ffmpeg`. The organization of datasets should be like this:

```

├── datasets
│   ├── mega
│   ├── vidor
│   │   ├── annotations
|   |   |   ├── training
|   |   |   |   ├── 0000
|   |   |   |   ├── ...
|   |   |   |   └── 1203
|   |   |   └── validation
|   |   |       ├── 0001
|   |   |       ├── ...
|   |   |       └── 1203
|   |   ├── cache
|   |   |   ├── MEGA_mask_clip_VidOR_training
|   |   |   ├── MEGA_mask_clip_VidOR_validation
|   |   |   ├── MEGA_mask_VidOR_training
|   |   |   └── MEGA_mask_VidOR_validation
|   |   ├── features
|   |   |   ├── clip_features_training
|   |   |   ├── clip_val_proposal
|   |   |   ├── GT_boxfeatures_training
|   |   |   ├── MEGA_VidORval_cache
|   |   |   |   └─ MEGAv9_m60s0.3_freq1_VidORval_freq1_th_15-180-200-0.40.pkl
|   |   |   └── vidor_per_video_val
|   |   ├── frames
│   │   └── videos
|   |       ├── 0000
|   |       ├── ...
|   |       └── 1203
|   ├── VidSGG-BIG 
|   ├── vidor_policy.txt
|   └── vidor_video_to_frames.py
├── models
...
├── VidVRD_helper
...
```


### VidOR

1. Download the [VidOR](https://xdshang.github.io/docs/vidor.html) dataset, unzip all videos (training and validation) into `datasets/vidor/videos`. Unzip the training and validation annotations into `datasets/vidor/annotations`.
2. Go to the `datasets` directory and run the following command to decode the videos into frames.
   ```
   python vidor_video_to_frames.py
   ```
3. Extract visual features from gt bounding boxes. We follow the Gao's method from https://github.com/Dawn-LX/VidVRD-tracklets. First, download [the pretrained weight of MEGA](https://drive.google.com/file/d/1nypbyRLpiQkxr7jvnnM4LEx2ZJuzrjws/view) and put it into `datasets/mega/ckpts`. Step into `datasets/mega` and run the following command to extract features.
   ```
   bash scripts/extract_vidor_gt.sh [gpu_id]
   ```
   Or we can download our extracted features from [Hugging Face](https://huggingface.co/datasets/guacamole99/vrdone_features) and unzip it into the corresponding place.
4. Download the [extracted proposal features of validation set](https://mega.nz/folder/VcwA1DaI#YW2M_uFsbsE6twHDIpfPuw) from [Gao's method (BIG)](https://github.com/Dawn-LX/VidSGG-BIG). Then, put it into `datasets/vidor/features/MEGA_VidORval_cache`. We copy the `dataloader` part from [BIG](https://github.com/Dawn-LX/VidSGG-BIG). Step into `datasets/VidSGG-BIG` and divide the proposal features into per-video ones by the following command:
   ```
   python prepare_vidor_proposal.py
   ```
5. [Optional] For the vrdone-x, we need to extract the clip features. First, step into the repository of [CLIP](https://github.com/openai/CLIP) and install it. Second, extract the features for training and evaluation respectively. We step into `datasets/mega`, run
   ```
   bash extract_vidor_clip_gt.sh [gpu_id]
   ```
   and
   ```
   bash extract_vidor_clip_val.sh [gpu_id]
   ```
   Or we can download our extracted features from [Hugging Face](https://huggingface.co/datasets/guacamole99/vrdone_features) and unzip them into the corresponding places.

### VidVRD

Coming soon ...

## Train 
1. **VrdONE for vidor**. Train with a single gpu.
   ```
   bash scripts/train_vidor.sh 1 [gpu_id]
   ```
   Change the `batch_size` in the config and train the model with multiple gpus.
   ```
   bash scripts/train_vidor.sh [num_gpus] [gpu_ids]
   ```
1. **VrdONE-X for vidor**. Train with a single gpu.
   ```
   bash scripts/train_vidor_x.sh 1 [gpu_id]
   ```
   Change the `batch_size` in the config and train the model with multiple gpus.
   ```
   bash scripts/train_vidor_x.sh [num_gpus] [gpu_ids]
   ```

## Eval
1. **VrdONE for vidor**. Download our vrdone checkpoint and evaluate with the following command.
   ```
   bash scripts/eval_vidor_ckpt.sh [gpu_id]
   ```
   If we trained our own model, we can evaluate all trained checkpoints.
   ```
   bash scripts/eval_vidor_multiple.sh [gpu_id]
   ```
   We can change the `topk` and the `ckpt_path` to check.

2. **VrdONE-X for vidor**. Download our vrdone-x checkpoint and evaluate with the following command.
   ```
   bash scripts/eval_vidor_x_ckpt.sh [gpu_id]
   ```
   If we trained our own model, we can evaluate all trained checkpoints.
   ```
   bash scripts/eval_vidor_x_multiple.sh [gpu_id]
   ```
   We can change the `topk` and the `ckpt_path` to check.

## Model Zoo
Hugging Face model repository: https://huggingface.co/guacamole99/vrdone

Hugging Face dataset repository: https://huggingface.co/datasets/guacamole99/vrdone_features

| Model | Dataset | Extra Features | Download Path |
|-------|---------------|-------| ---------------|
|VrdONE | VidOR | - | [Hugging Face](https://huggingface.co/guacamole99/vrdone)|
|VrdONE-X | VidOR | CLIP | [Hugging Face](https://huggingface.co/guacamole99/vrdone) |
|VrdONE | ImageNet-VidVRD | - | |

## Citation
```
@inproceedings{jiang2024vrdone,
  author = {Jiang, Xinjie and Zheng, Chenxi and Xu, Xuemiao and Liu, Bangzhen and Zheng, Weiying and Zhang, Huaidong and He, Shengfeng},
  title = {VrdONE: One-stage Video Visual Relation Detection},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  year = {2024},
}
```

## Acknowledgement

This project is mainly based on [ActionFormer](https://github.com/happyharrycn/actionformer_release), [MaskFormer](https://github.com/facebookresearch/MaskFormer), and [BIG](https://github.com/Dawn-LX/VidSGG-BIG). Thanks for their amazing projects!
