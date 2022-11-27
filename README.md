# Zero Shot Image Restoration Using Denoising Diffusion Null-Space Model

![image](https://user-images.githubusercontent.com/95485229/198285474-ff2e43de-9fc5-40c4-840b-f902bac4fa3c.png)

## Brief
This repository contains the code release for ***Zero Shot Image Restoration Using Denoising Diffusion Null-Space Model ([DDNM](https://openreview.net/forum?id=mRieQgMtNTQ))***.

***Supported Applications:***
- Super-Resolution
- Colorization
- Inpainting
- Deblurring
- Compressed Sensing
- Old Photo Restoration🌟


## Installation
`git clone https://github.com/wyhuai/DDNM.git; cd nerfocus`  
`conda install pip; pip install --upgrade pip`  
`pip install -r requirements.txt`  

## Evaluation
We provide a pretrained model in experiments/horns, so you can run the following command to generate a video with defocus effects. You may change the lens parameters "l" and "a" in eval_vid.py to adjust the focus distance and aperture size. 
`python -m eval_vid --data_dir=horns --train_dir=experiments/horns --chunk=3196 --gin_file=configs/llff.gin --logtostderr`

## Try DDNM on Other Diffusion Models
It is ***very easy*** to implement a basic DDNM on your own diffusion model! You may reference the following code:
1. Copy these operator implementations to the core diffusion sampling file.
```python
def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return th.stack((x*coef/base, x*coef/base, x*coef/base), 1)    
    
def PatchUpsample(x, scale):
    n, c, h, w = x.shape
    x = torch.zeros(n,c,h,scale,w,scale) + x.view(n,c,h,1,w,1)
    return x.view(n,c,scale*h,scale*w)

# Implementation of A and its pseudo-inverse Ap    
    
if IR_mode=="colorization":
    A = color2gray
    Ap = gray2color
    
elif IR_mode=="inpainting":
    A = lambda z: z*mask
    Ap = A
      
elif IR_mode=="super resolution":
    A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
    Ap = lambda z: PatchUpsample(z, scale)

elif IR_mode=="old photo restoration":
    A1 = lambda z: z*mask
    A1p = A1
    
    A2 = color2gray
    A2p = gray2color
    
    A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
    A3p = lambda z: PatchUpsample(z, scale)
    
    A = lambda z: A3(A2(A1(z)))
    Ap = lambda z: A1p(A2p(A3p(z)))
```
1. Find the variant $\mathbf{x}\_{0|t}$ in the target codes, using the result of this function to modify the sampling of $\mathbf{x}\_{t-1}$.
```python
# Core Implementation of DDNM+, simplified denoising solution

def ddnmp_core(x0t, y, sigma_y, sigma_t, a_t):

    #Eq 19
    if sigma_t >= a_t*sigma_y: 
        lambda_t = 1
        gamma_t = sigma_t**2 - (a_t*lambda_t*sigma_y)**2
    else:
        lambda_t = sigma_t/(a_t*sigma_y)
        gamma_t = 0
        
    #Eq 17    
    x0t= x0t + lambda_t*Ap(y - A(x0t))
    
    return x0t, gamma_t
```





