# Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model
## 📖[**Paper**](https://arxiv.org/pdf/2212.00490.pdf)|🖼️[**Project Page**](https://wyhuai.github.io/ddnm.io/)| <a href="https://colab.research.google.com/drive/1SRSD6GXGqU0eO2CoTNY-2WykB9qRZHJv?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/DDNM-HQ)

[Yinhuai Wang*](https://wyhuai.github.io/info/), [Jiwen Yu*](https://scholar.google.com.hk/citations?user=uoRPLHIAAAAJ), [Jian Zhang](https://jianzhang.tech/)  
Peking University and PCL  
\*denotes equal contribution



This repository contains the code release for *Zero-Shot Image Restoration Using ***D***enoising ***D***iffusion ***N***ull-Space ***M***odel*. **DDNM** can solve various image restoration tasks **without any optimization or training! Yes, in a zero-shot manner**.


***Supported Applications:***
- **Arbitrary Size**🆕
- **Old Photo Restoration**🆕
- Super-Resolution
- Denoising
- Colorization
- Inpainting
- Deblurring
- Compressed Sensing


![front](https://user-images.githubusercontent.com/95485229/227095293-1024f337-1fde-494b-ae82-97d6139bbefe.jpg)



## 🧩News
- A Colab demo for high-quality results is now avaliable! <a href="https://colab.research.google.com/drive/1SRSD6GXGqU0eO2CoTNY-2WykB9qRZHJv?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>

## Installation
### Code
```
git clone https://github.com/wyhuai/DDNM.git
```
### Environment
```
pip install numpy torch blobfile tqdm pyYaml pillow    # e.g. torch 1.7.1+cu110.
```
### Pre-Trained Models
To restore human face images, download this [model](https://drive.google.com/file/d/1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21/view?usp=share_link)(from [SDEdit](https://github.com/ermongroup/SDEdit)) and put it into `DDNM/exp/logs/celeba/`. 
```
https://drive.google.com/file/d/1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21/view?usp=share_link
```
To restore general images, download this [model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)(from [guided-diffusion](https://github.com/openai/guided-diffusion)) and put it into `DDNM/exp/logs/imagenet/`.
```
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```
### Quick Start
Run below command to get 4x SR results immediately. The results should be in `DDNM/exp/image_samples/demo`.
```
python main.py --ni --simplified --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0 -i demo
```

## Setting
The detailed sampling command is here:
```
python main.py --ni --simplified --config {CONFIG}.yml --path_y {PATH_Y} --eta {ETA} --deg {DEGRADATION} --deg_scale {DEGRADATION_SCALE} --sigma_y {SIGMA_Y} -i {IMAGE_FOLDER}
```
with following options:
- We implement **TWO** versions of DDNM in this repository. One is SVD-based version, which is more precise in solving noisy tasks. Another one is the simplified version, which does not involve SVD and is flexible for users to define their own degradations. Use `--simplified` to activate the simplified DDNM. Without `--simplified` will turn to the SVD-based DDNM.
- `PATH_Y` is the folder name of the test dataset, in `DDNM/exp/datasets`.
- `ETA` is the DDIM hyperparameter. (default: `0.85`)
- `DEGREDATION` is the supported tasks including `cs_walshhadamard`, `cs_blockbased`, `inpainting`, `denoising`, `deblur_uni`, `deblur_gauss`, `deblur_aniso`, `sr_averagepooling`,`sr_bicubic`, `colorization`, `mask_color_sr`, and user-defined `diy`.
- `DEGRADATION_SCALE` is the scale of degredation. e.g., `--deg sr_bicubic --deg_scale 4` lead to 4xSR.
- `SIGMA_Y` is the noise observed in y.
- `CONFIG` is the name of the config file (see `configs/` for a list), including hyperparameters such as batch size and sampling step.
- `IMAGE_FOLDER` is the folder name of the results.

For the config files, e.g., celeba_hq.yml, you may change following properties:
```
sampling:
    batch_size: 1
    
time_travel:
    T_sampling: 100     # sampling steps
    travel_length: 1    # time-travel parameters l and s, see section 3.3 of the paper.
    travel_repeat: 1    # time-travel parameter r, see section 3.3 of the paper.
```

## Reproduce The Results In The Paper
### Quantitative Evaluation
Dataset download link: [[Google drive](https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4?usp=sharing)] [[Baidu drive](https://pan.baidu.com/s/1tQaWBqIhE671v3rrB-Z2mQ?pwd=twq0)]

Download the CelebA testset and put it into `DDNM/exp/datasets/celeba/`.

Download the ImageNet testset and put it into `DDNM/exp/datasets/imagenet/` and replace the file `DDNM/exp/imagenet_val_1k.txt`.

Run the following command. You may increase the batch_size to accelerate evaluation.
```
sh evaluation.sh
```

### High-Quality Results
You can try this [**Colab demo**](https://colab.research.google.com/drive/1SRSD6GXGqU0eO2CoTNY-2WykB9qRZHJv?usp=sharing) for High-Quality results. Note that the High-Quality results presented in the front figure are mostly generated by applying DDNM to the models in [RePaint](https://github.com/andreas128/RePaint).

## 🔥Real-World Applications
### Demo: Real-World Super-Resolution.
![orig_62](https://user-images.githubusercontent.com/95485229/204471148-bf155c60-c7b3-4c3a-898c-859cb9d0d723.png)
![00000](https://user-images.githubusercontent.com/95485229/204971948-7564b536-b562-4187-9d8c-d96db4c55f7c.png)

Run the following command
```
python main.py --ni --simplified --config celeba_hq.yml --path_y solvay --eta 0.85 --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0.1 -i demo
```
### Demo: Old Photo Restoration.
![image](https://user-images.githubusercontent.com/95485229/204973149-4818426b-89af-410c-b1b7-f26b8f65358b.png)
![image](https://user-images.githubusercontent.com/95485229/204973288-0f245e93-8980-4a32-a5e9-7f2bfe58d8eb.png)

Run the following command
```
python main.py --ni --simplified --config oldphoto.yml --path_y oldphoto --eta 0.85 --deg "mask_color_sr" --deg_scale 2.0 --sigma_y 0.02 -i demo
```
# References
If you find this repository useful for your research, please cite the following work.
```
@article{wang2022zero,
  title={Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model},
  author={Wang, Yinhuai and Yu, Jiwen and Zhang, Jian},
  journal={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```
This implementation is based on / inspired by:
- https://github.com/wyhuai/RND (null-space learning)
- https://github.com/andreas128/RePaint (time-travel trick)
- https://github.com/bahjat-kawar/ddrm (code structure)
