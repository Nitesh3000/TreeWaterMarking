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
from scipy.fftpack import dct
import torch_dct as dct
from torchvision.transforms import ToTensor
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



def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0)**2 + (y-y0)**2)<= r**2
# Returns a binary mask tensor of the same shape as init_latents_w, with specified watermarking regions set to True.
def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # Apply to all channels
            watermarking_mask[:, :] = torch_mask
        else:
            # Apply to a specific channel
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # Apply to all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            # Apply to a specific channel
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=None):
    set_random_seed(args.w_seed)
    
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    # Apply DCT if needed
    if any(pattern in args.w_pattern for pattern in ['rand', 'zeros', 'const', 'ring']):
        gt_init = gt_init.to(torch.float32)
        gt_init = dct.dct_2d(gt_init,norm = "ortho")
        gt_init = gt_init.to(torch.float16)

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'const' in args.w_pattern:
        gt_patch = gt_init * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = gt_init
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch

def readImage(image_path,args,i):
    image = Image.open(image_path).convert('RGB')
    save_image_to_unique_folder(image, args.output_dir, i,"apple_image_for_watermarking")
    
    image_tensor = ToTensor()(image)
    # if image_tensor.shape[0] == 4:
    #     image_tensor = image_tensor[:3, :, :]
    save_image_to_unique_folder(image_tensor, args.output_dir, i,"apple_image_for_watermarking_tensor")
    # image_tensor = torch.tensor(np.array(image)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(torch.float16)
    # image_tensor = image_tensor.expand([1,3,475,584])
    return image_tensor

def inject_watermark(init_latents_w, image, args, pipe, device,i):
    # Obtain the latent representation of the image
    image_latents = pipe.get_image_latents(image.to(device), sample=False)
    save_image_to_unique_folder(image_latents, args.output_dir, i,"imageforwatermarkingLatents")
    image_latents = image_latents.to(torch.float32)
    watermark_size = (32, 32)  # Size of the watermark to be inserted

        # Resize the watermark
    image_latents = F.interpolate(image_latents, size=watermark_size, mode='bilinear', align_corners=False)
    # Perform DCT on the image latents
    dct_image = dct.dct_2d(image_latents)
    dct_image = dct_image.to(torch.float16)
    save_image_to_unique_folder(dct_image, args.output_dir, i,"imageforwatermarkingLatents_DCT")
    
    if args.w_injection == 'complex':
        init_latents_w = init_latents_w.clone()
        save_image_to_unique_folder(init_latents_w, args.output_dir, i,"init_latents_w_not_watermarked")
        init_latents_w = init_latents_w.to(torch.float32)
        
        # Perform DCT on the initial latents
        init_latents_w_dct = dct.dct_2d(init_latents_w)
        init_latents_w_dct = init_latents_w_dct.to(torch.float16)
        save_image_to_unique_folder(init_latents_w_dct, args.output_dir, i,"init_latents_w_dct_not_watermarked")
        
        watermark_scale = 0.2
        # watermark_size = (32, 32)  # Size of the watermark to be inserted

        # Resize the watermark
        # dct_image_resized = F.interpolate(dct_image, size=watermark_size, mode='bilinear', align_corners=False)

        # Define the position to place the watermark (center of the latent image)
        h, w = init_latents_w_dct.shape[-2:]
        center_y, center_x = h // 2, w // 2
        half_watermark_h, half_watermark_w = watermark_size[0] // 2, watermark_size[1] // 2

        # Add the watermark to the center of the existing latents
        init_latents_w_dct[:, :, center_y - half_watermark_h:center_y + half_watermark_h,
                           center_x - half_watermark_w:center_x + half_watermark_w] += (
            watermark_scale * dct_image
        )
        save_image_to_unique_folder(init_latents_w_dct, args.output_dir, i,"init_latents_w_dct after watermark injection")

        # Perform IDCT to get back to the image space
        init_latents_w = init_latents_w.to(torch.float32)
        init_latents_w = dct.idct_2d(init_latents_w_dct)
        init_latents_w = init_latents_w.to(torch.float16)

    else:
        raise NotImplementedError(f'w_injection: {args.w_injection}')
    
    return init_latents_w, watermark_size


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermark_size,init_latents, args):
    # Perform DCT on the reversed latents if necessary
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_dct = dct.dct_2d(reversed_latents_no_w)
        reversed_latents_w_dct = dct.dct_2d(reversed_latents_w)
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_dct = reversed_latents_no_w
        reversed_latents_w_dct = reversed_latents_w
    else:
        raise NotImplementedError(f'w_measurement: {args.w_measurement}')
    
    # Ensure tensors are in float32 for precision
    reversed_latents_no_w_dct = reversed_latents_no_w_dct.to(torch.float32)
    reversed_latents_w_dct = reversed_latents_w_dct.to(torch.float32)

    # Define the watermark size
    watermark_h, watermark_w = watermark_size

    # Get the center position for the watermark
    h, w = reversed_latents_no_w_dct.shape[-2:]
    center_y, center_x = h // 2, w // 2
    half_watermark_h, half_watermark_w = watermark_h // 2, watermark_w // 2

    # Define the ROI for watermark comparison
    no_w_roi = reversed_latents_no_w_dct[:, :, center_y - half_watermark_h:center_y + half_watermark_h,
                                       center_x - half_watermark_w:center_x + half_watermark_w]
    w_roi = reversed_latents_w_dct[:, :, center_y - half_watermark_h:center_y + half_watermark_h,
                                   center_x - half_watermark_w:center_x + half_watermark_w]
    original_roi = init_latents[:, :, center_y - half_watermark_h:center_y + half_watermark_h,
                                       center_x - half_watermark_w:center_x + half_watermark_w]

    # Compute L1 metrics for the region of interest
    no_w_metric = torch.abs(no_w_roi - original_roi).mean().item()
    w_metric = torch.abs(w_roi - no_w_roi).mean().item()  # Metric of change due to watermarking
    
    return no_w_metric, w_metric

def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w
def save_image_to_unique_folder(image, base_dir, index,name):
    # Define folder path based on index
    folder_path = os.path.join(base_dir, f"extraction/AppleGrayscaleWatermarkedWithImageBlurring/{index}")

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Define the file path
    file_path = os.path.join(folder_path, f"{name}.png")

    if isinstance(image, Image.Image):
        # If input is a PIL Image, save it directly
        image.save(file_path)
    elif isinstance(image, torch.Tensor):
        # If input is a tensor, convert it to an image and save
        save_image(image, file_path)
    else:
        raise TypeError("Input must be of type PIL.Image or torch.Tensor")

def saveRocCurve(fpr, tpr, base_dir, index):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Convert the plot to a tensor
    plt_canvas = plt.gcf().canvas
    plt_canvas.draw()
    roc_image = np.array(plt_canvas.renderer.buffer_rgba())
    roc_tensor = ToTensor()(roc_image)
    
    # Save the ROC curve using the provided function
    save_image_to_unique_folder(roc_tensor, base_dir, index, "roc_curve")
    
    # Close the plot
    plt.close()

def extract_watermark(watermarked_latents, original_latents, watermark_size, pipe, device,args,i):
    # Ensure the latents are in the correct precision
    watermarked_latents = watermarked_latents.to(torch.float32)
    original_latents = original_latents.to(torch.float32)
    
    # Perform DCT on the watermarked and original latents
    watermarked_latents_dct = dct.dct_2d(watermarked_latents)
    original_latents_dct = dct.dct_2d(original_latents)
    watermarked_latents_dct = watermarked_latents_dct.to(torch.float16)
    original_latents_dct = original_latents_dct.to(torch.float16)
    # Calculate the difference to extract the watermark DCT coefficients
    watermark_dct = (watermarked_latents_dct - original_latents_dct)/0.2
    
    # Define the position of the watermark in the latent space
    h, w = watermarked_latents_dct.shape[-2:]
    print(h,w)
    center_y, center_x = h // 2, w // 2
    half_watermark_h, half_watermark_w = watermark_size[0] // 2, watermark_size[1] // 2
    
    # Extract the watermark from the center of the latent space
    extracted_watermark_dct = watermark_dct[:, :, center_y - half_watermark_h:center_y + half_watermark_h,
                                            center_x - half_watermark_w:center_x + half_watermark_w]
    save_image_to_unique_folder(extracted_watermark_dct, args.output_dir, i, "extracted_watermark_dct")
    # Resize the extracted watermark back to the original image size if needed
    # extracted_watermark_dct = F.interpolate(extracted_watermark_dct, size=(64, 64), mode='bilinear', align_corners=False)
    
    # Perform IDCT to recover the watermark in the image space
    extracted_watermark_dct = extracted_watermark_dct.to(torch.float32)
    extracted_watermark = dct.idct_2d(extracted_watermark_dct)
    extracted_watermark = extracted_watermark.to(torch.float16)
    return extracted_watermark