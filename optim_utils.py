import torch
from torchvision import transforms
# pip install datasets
from datasets import load_dataset

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy
import os
from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.fft as fft
from torchvision.utils import save_image
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import pytorch_wavelets as pw
import torch.nn.functional as F
def create_directories(base_dir, image_index):
    """Create directories for saving images."""
    dirs = {
        'original': os.path.join(base_dir, f'image_{image_index}'),
        'distorted': os.path.join(base_dir, f'image_{image_index}'),
        'watermarked': os.path.join(base_dir, f'image_{image_index}'),
        'fourier_before': os.path.join(base_dir, f'image_{image_index}'),
        'fourier_after': os.path.join(base_dir, f'image_{image_index}'),
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs





def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    #scales the pixel values from the range [0, 1] to the range [-1, 1].
    return 2.0 * image - 1.0 #it can improve the convergence of training algorithms.


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img2)
        
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key


def circle_mask(size=64, r=10, x_offset=0, y_offset=0): #True inside the circle, False elseWhere
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2 #center of the grid
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2

def get_watermarking_pattern(pipe, args, device, shape=None):
    dtcwt_forward = DTCWTForward(J=5, biort='near_sym_b', qshift='qshift_b').to(device)
    dtcwt_inverse = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(device)
    
    set_random_seed(args.w_seed)
    
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init.clone()

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            # Resize mask to match gt_patch dimensions
            tmp_mask_resized = F.interpolate(tmp_mask.unsqueeze(0).unsqueeze(0).float(), 
                                             size=(gt_patch.shape[-2], gt_patch.shape[-1]), 
                                             mode='bilinear', align_corners=False).bool().squeeze()

            for j in range(gt_patch.shape[1]):
                # Use proper indexing and mask assignment
                gt_patch[:, j] = torch.where(tmp_mask_resized, gt_patch_tmp[:, j], gt_patch[:, j])

    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    
    elif 'rand' in args.w_pattern:
        Yl, Yh = dtcwt_forward(gt_init)
        gt_patch = Yl
        gt_patch[:] = gt_patch[0]
    
    elif 'zeros' in args.w_pattern:
        Yl, Yh = dtcwt_forward(gt_init)
        Yl *= 0
        Yh = tuple(coeff * 0 for coeff in Yh)
        gt_patch = Yl
    
    elif 'const' in args.w_pattern:
        Yl, Yh = dtcwt_forward(gt_init)
        Yl *= 0
        gt_patch = Yl + args.w_pattern_const
    
    elif 'ring' in args.w_pattern:
        Yl, Yh = dtcwt_forward(gt_init)
        gt_patch = Yl.clone()
        # gt_patch_tmp = copy.deepcopy(Yl)
        gt_patch_tmp = Yl.clone()
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i) #shape of the latent
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            # Resize mask to match gt_patch dimensions
            #interpolate happens on the 4d tensor
            tmp_mask_resized = F.interpolate(tmp_mask.unsqueeze(0).unsqueeze(0).float(), 
                                             size=(gt_patch.shape[-2], gt_patch.shape[-1]), 
                                             mode='bilinear', align_corners=False).bool().squeeze() #bilinear works for 2d image with 4d tensor

            # Ensure the mask and tensors have the same dimensions
            if gt_patch.shape[2:] == tmp_mask_resized.shape:
                gt_patch[:, :, tmp_mask_resized] = gt_patch_tmp[:, :, tmp_mask_resized] #batch, channel, height,width

    # print("type of gt_patch: ", gt_patch.type())
    # resized_mask = F.interpolate(gt_patch.float(), size=(16, 16), mode='bilinear', align_corners=False)
    return gt_patch


# Returns a binary mask tensor of the same shape as init_latents_w, with specified watermarking regions set to True.
def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device) 

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask  #spatial dimensions and channels
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')
    resized_mask = F.interpolate(watermarking_mask.float(), size=(16, 16), mode='bilinear', align_corners=False)
    resized_mask = resized_mask > 0.5
    return resized_mask



def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args, i, device):
    # Perform Dual-Tree Complex Wavelet Transform (DTCWT)
    dwt = DTCWTForward(J=5, biort='near_sym_b', qshift='qshift_b').to(device)
    dtcwt_inverse = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(device)
    
    # Apply DTCWT and get the coefficients
    init_latents_w_dwt, yh = dwt(init_latents_w)
    
    # Resize watermarking_mask to match the low-frequency component size
    resized_mask = F.interpolate(watermarking_mask.float(), size=init_latents_w_dwt.shape[-2:], mode='bilinear', align_corners=False).bool().to(device)
    
    # Resize gt_patch to match the low-frequency component size
    resized_gt_patch = F.interpolate(gt_patch.float(), size=init_latents_w_dwt.shape[-2:], mode='bilinear', align_corners=False).to(device)

    # Ensure data types match
    if init_latents_w_dwt.dtype != resized_gt_patch.dtype:
        resized_gt_patch = resized_gt_patch.to(init_latents_w_dwt.dtype)
    
    save_image(init_latents_w[0], f"{args.output_dir}/image_{i}/latent_no_w_image.png")
    save_image(init_latents_w_dwt[0, 0].real, f"{args.output_dir}/image_{i}/dwt_no_w_image.png")  # Save the real part of the low-frequency component

    # Inject watermark into the middle frequency components
    if len(yh) > 2:  # Ensure there are at least three frequency bands (low, middle, high)
        # Process middle frequency components, excluding the first (low) and last (high) components
        for j in range(1, len(yh) - 1):
            # Resize the watermark mask and patch to match the current frequency component size
            middle_freq_mask = F.interpolate(watermarking_mask.float(), size=yh[j].shape[-3:-1], mode='bilinear', align_corners=False).bool().to(device)
            resized_gt_patch_middle = F.interpolate(gt_patch.float(), size=yh[j].shape[-3:-1], mode='bilinear', align_corners=False).to(device)

            # Ensure data types match
            if yh[j].dtype != resized_gt_patch_middle.dtype:
                resized_gt_patch_middle = resized_gt_patch_middle.to(yh[j].dtype)

            # Adjust dimensions for broadcasting
            middle_freq_mask = middle_freq_mask.unsqueeze(2).unsqueeze(-1)  # Add dimensions for directions and real/imaginary
            resized_gt_patch_middle = resized_gt_patch_middle.unsqueeze(2).unsqueeze(-1)  # Add dimensions for directions and real/imaginary

            # Broadcast mask and watermark patch to match the shape of yh[j]
            middle_freq_mask = middle_freq_mask.expand(-1, -1, yh[j].shape[2], -1, -1, yh[j].shape[-1])
            resized_gt_patch_middle = resized_gt_patch_middle.expand(-1, -1, yh[j].shape[2], -1, -1, yh[j].shape[-1])

            if args.w_injection == 'complex':
                yh[j][middle_freq_mask] = resized_gt_patch_middle[middle_freq_mask].clone()
            elif args.w_injection == 'seed':
                yh[j] = resized_gt_patch_middle.clone()
            else:
                raise NotImplementedError(f'w_injection: {args.w_injection}')

            yh_real_part = yh[j][0, :, 0, :, :, 0].real  # Extract a 2D slice that can be saved as an image
            yh_real_part = (yh_real_part - yh_real_part.min()) / (yh_real_part.max() - yh_real_part.min())  # Normalize to [0, 1]
            save_image(yh_real_part, f"{args.output_dir}/image_{i}/yh_w_image_middle_{j}.png")

    # Perform Inverse Dual-Tree Complex Wavelet Transform (IDTCWT)
    init_latents_w = dtcwt_inverse((init_latents_w_dwt, yh))  # Apply IDWT to get the image back

    save_image(init_latents_w[0], f"{args.output_dir}/image_{i}/latent_w_image.png")
    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args, device):
    # Initialize the DTCWT and its inverse
    dwt = DTCWTForward(J=5, biort='near_sym_b', qshift='qshift_b').to(device)
    dtcwt_inverse = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(device)
    
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_dwt, yh_no_w = dwt(reversed_latents_no_w)
        reversed_latents_w_dwt, yh_w = dwt(reversed_latents_w)
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_dwt = reversed_latents_no_w
        reversed_latents_w_dwt = reversed_latents_w
        target_patch = gt_patch
    else:
        raise NotImplementedError(f'w_measurement: {args.w_measurement}')

    # Resize watermarking_mask to match the dimensions of the DWT coefficients
    resized_mask = F.interpolate(watermarking_mask.float(), size=reversed_latents_no_w_dwt.shape[-2:], mode='bilinear', align_corners=False).bool().to(device)
    
    if 'l1' in args.w_measurement:
        # Ensure target_patch is resized to match the DWT coefficient dimensions
        resized_target_patch = F.interpolate(target_patch.float(), size=reversed_latents_no_w_dwt.shape[-2:], mode='bilinear', align_corners=False).to(device)

        # Evaluate the watermark using L1 norm
        no_w_metric = torch.abs(reversed_latents_no_w_dwt[resized_mask] - resized_target_patch[resized_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_dwt[resized_mask] - resized_target_patch[resized_mask]).mean().item()
    else:
        raise NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric



def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_dwt = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_dwt = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_dwt = torch.concatenate([reversed_latents_no_w_dwt.real, reversed_latents_no_w_dwt.imag])
    sigma_no_w = reversed_latents_no_w_dwt.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_dwt - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_dwt = torch.concatenate([reversed_latents_w_dwt.real, reversed_latents_w_dwt.imag])
    sigma_w = reversed_latents_w_dwt.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_dwt - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w
