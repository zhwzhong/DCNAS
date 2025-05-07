## DCNAS: Dual-Level Cross-Modality Neural Architecture Search for Guided Image Super-Resolution

This repository provides the official implementation of our paper:

**Dual-Level Cross-Modality Neural Architecture Search for Guided Image Super-Resolution**  
Zhiwei Zhong, Xianming Liu, Junjun Jiang, Debin Zhao, and Shiqi Wang  
(*Under review at IEEE Transactions on Pattern Analysis and Machine Intelligence*) 
 

## 🔍 Abstract

Guided image super-resolution (GISR) aims to reconstruct a high-resolution (HR) target image from its low-resolution (LR) counterpart with the guidance of a HR image from another modality. Existing learning-based methods typically employ symmetric two-stream networks to extract features from both the guidance and target images, and then fuse these features at either an early or late stage through manually designed modules to facilitate joint inference. Despite significant performance, these methods still face several issues: i) the symmetric architectures treat images from different modalities equally, which may overlook the inherent differences between them; ii) lower-level features contain detailed information while higher-level features capture semantic structures. However, determining which layers should be fused and which fusion operations should be selected remain unresolved; iii) most methods achieve performance gains at the cost of increased computational complexity, so balancing the trade-off between computational complexity and model performance remains a critical issue. To address these issues, we propose a Dual-level Cross-modality Neural Architecture Search (DCNAS) framework to automatically design efficient GISR models. Specifically, we propose a dual-level search space that enables the NAS algorithm to identify effective architectures and optimal fusion strategies. Moreover, we propose a supernet training strategy that employs a pairwise ranking loss trained performance predictor to guide the supernet training process. To the best of our knowledge, this is the first attempt to introduce the NAS algorithm into GISR tasks. Extensive experiments demonstrate that the discovered model family, DCNAS-Tiny and DCNAS, achieve significant improvements on several GISR tasks, including guided depth map super-resolution, guided saliency map super-resolution, guided thermal image super-resolution, and pan-sharpening. Furthermore, we analyze the architectures searched by our method and provide some new insights for future research.


<p align="center">
  <img src="https://github.com/zhwzhong/DCNAS/blob/main/network.png">
</p>


## :bar_chart: Results

<p align="center">
  <img src="https://github.com/zhwzhong/DCNAS/blob/main/res.png">
</p>


## :wrench: Dependencies

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0 (https://pytorch.org/
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## :hammer: Installation

1. Clone repo

   ```bash
   git https://github.com/zhwzhong/DCNAS.git
   cd DCNAS
   ```

2. Install dependent packages

   ```bash
   pip install -r requirements.txt
   ```

## :package: Dataset

- Guided Depth Map SR:

  > For this task, we use two widely used benchmark datasets: the NYU v2 dataset
and the RGB-D-D dataset. The NYU v2 dataset is a large scale indoor 
dataset containing 1,449 RGB-D image pairs. We use the first 1,000 image pairs
as the training set and the remaining 449 image pairs as the testing set. To 
verify the generalization ability of the proposed method, we further incorporate five additional
datasets into our evaluation: 1) 1,064 RGB-D pairs from Sintel dataset; 2) the testset
of DIDOE indoor dataset; 3) the first 500 RGB-D pairs from SUN RGBD testset; 4) the testset of RGB-D-D dataset; 5) the testset of DIML indoor dataset. 5) 
For RGB-D-D dataset, we use the official training and testing splits as the training and test sets.
Download Link: 1. [NYU 2. ](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)[Sintel](http://sintel.is.tue.mpg.de/) 3.[DIDOE](https://github.com/diode-dataset/diode-devkit?utm_source=catalyzex.com) 4. [SUN RGB](https://rgbd.cs.princeton.edu/) 5. [RGB-D-D](https://github.com/lingzhi96/RGB-D-D-Dataset) 6. [DIML](https://dimlrgbd.github.io/)
- Guided Saliency Map SR:

   > For this task, we utilize the [DUT-OMRON](https://saliencydetection.net/dut-omron/) dataset as the testing set and employ bicubic 
downsampling with a scale factor of 8 to generate LR saliency maps.

- Guided Thermal Image SR:

   > For this task, we use the training set provided by [this work](https://openaccess.thecvf.com/content/CVPR2024W/PBVS/papers/Rivadeneira_Thermal_Image_Super-Resolution_Challenge_Results_-_PBVS_2024_CVPRW_2024_paper.pdf)  as our training set.
Since the authors do not provide ground-truth images for their testing set, we employ the
validation set as the testing set and randomly select 100 image pairs from the training
set to serve as our validation set.

- Pansharpening:
   
   > We employ the dataset provided by [this work](https://ieeexplore.ieee.org/document/10890746/) as the training and testing dataset.

## :rocket: Train

You can also train by yourself:

```
torchrun --nnodes 1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:12343 main.py --train_supernet --model MODEL_NAME --dataset DATA_NAME --scale SCALE
```

**Common options:**

| Argument           | Description                       | Example            |
|--------------------|-----------------------------------|--------------------|
| `--train_supernet` | supernet training                 | `--train_supernet` |
| `--search`         | architecture search               | `--search`         |
| `--train_random`   | train the search network          | `--train_random`   |
| `--num_blocks`     | number of blocks for each stage   | `--num_blocks 4`   |
| `--num_stages`     | number of down/up sampling stages | `--num_stages 4`   |
| `--num_features`   | feature channels                  | `--num_features 8` |
| `--model`          | model name                        | `--model NAME`     |
| `--scale`          | Super-resolution upscale factor   | `--scale=16`       |
| `--batch_size`     | Training batch size               | `--batch_size=16`  |
| `--lr`             | Initial learning rate             | `--lr=1e-4`        |
| `--epochs`         | Number of training epochs         | `--epochs=200`     |
| `--dataset`        | dataset name                      | `--dataset=NYU`    |
| `...`              | ...                               | `...`              |


## :rocket: Search

```
python main.py --search --model MODEL_NAME --dataset DATA_NAME
```


## :rocket: re-train

```
python main.py --train_random --model MODEL_NAME --dataset DATA_NAME
```

## :rocket: Test

We provide the pre-trained models in [[Model Zoo]](https://pan.baidu.com/s/1321_Z5CODC8qZs-10zwTNA?pwd=GISR)
With the trained model,  you can test your images.

```
python main.py --test_only --model MODEL_NAME --dataset DATA_NAME
```

## :gear: Acknowledgments

Thanks the editors and the reviewers for their insightful comments, which are very helpful to improve our paper!


## :memo: Citation
```
@unpublished{zhong2024dcnas,
  title={Dual-Level Cross-Modality Neural Architecture Search for Guided Image Super-Resolution},
  author={Zhiwei Zhong, Xianming Liu, Junjun Jiang, Debin Zhao, and Shiqi Wang },
  note={Under review at IEEE TPAMI},
  year={2024}
}
```