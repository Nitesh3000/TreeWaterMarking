import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import torch.nn as nn
import torch
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
from torchvision.utils import save_image
import config
import sys
import os
from sklearn import metrics
import torch_dct as dct
import torch_dct
from pycocotools.coco import COCO
from torch.nn.functional import mse_loss
from torchvision.models import vgg19
from torchvision import datasets, transforms
import zipfile
from ENCODERDECODER import VAE
import certifi
import sklearn.metrics as sk_metrics
from config import config
import matplotlib.pyplot as plt
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def extract_watermark(watermarked_latents, non_watermarked_latents, target_channel=0, extract_size=(32, 32), upscale_factor=1.0):
   
    _, num_channels, h, w = watermarked_latents.shape

    if extract_size[0] > h or extract_size[1] > w:
        raise ValueError(f"Extract size {extract_size} is too large for the tensor size: height={h}, width={w}.")

    # Calculate the center of the image to extract the watermark
    center_h, center_w = h // 2, w // 2
    start_h, start_w = center_h - extract_size[0] // 2, center_w - extract_size[1] // 2
    end_h, end_w = start_h + extract_size[0], start_w + extract_size[1]

    extracted_regions = []

   
    dct_latents_w = dct.dct_2d(watermarked_latents[:, target_channel:target_channel + 1, :, :])
    dct_latents_no_w = dct.dct_2d(non_watermarked_latents[:, target_channel:target_channel + 1, :, :])
    
    dct_latents_diff = dct_latents_w

         
    extracted_region_dct = dct_latents_diff[ :,:, start_h:end_h, start_w:end_w]
            
            
    extracted_region = dct.idct_2d(extracted_region_dct)
    

    return extracted_region
def evaluation_watermark(no_w_roi, w_roi, original_roi, args, i):
    
    
    
    no_w_roi_denorm = (no_w_roi + 1) / 2
    w_roi_denorm = (w_roi + 1) / 2
    original_roi_denorm = (original_roi + 1) / 2

   
    
    no_w_metric = torch.abs(no_w_roi_denorm - original_roi_denorm).mean().item()
    w_metric = torch.abs(w_roi_denorm - original_roi_denorm).mean().item()
    
    return no_w_metric, w_metric
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True).features[:16].eval()  # Use first few layers of VGG19
        for param in vgg.parameters():
            param.requires_grad = False
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.vgg = vgg.to(device)
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
def main(args):
    table = None
  
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    TRANSFORMATION = config.get("TRANSFORMATION", "dct")  
    os.makedirs(f"output_{TRANSFORMATION}", exist_ok=True)
    
   
    
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler', local_files_only=True)
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)
   
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    
    ])
  
    watermark_image = Image.open("/home/nitesh-mtech/tree-ring-watermark-main/OK.png")
   
   
    model = VAE(in_channels=1, latent_channels=1, out_channels=1)

    
    checkpoint_path = "/home/nitesh-mtech/tree-ring-watermark-main/output_images copy/vae_model3.pth"
    decoder_checkpoint_path = "/home/nitesh-mtech/tree-ring-watermark-main/output_dct_decoder_training/decoder_final.pth"
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    model = model.to(device)
    model.eval()  
    
    # Extract the encoder and decoder.
    encoder = model.encoder
    decoder = model.decoder
    decoder.load_state_dict(torch.load(decoder_checkpoint_path, map_location=device))
    decoder.to(device)
    decoder.eval()
    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)
    
    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)
   
    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []
    # Assume 'transform' is defined as:
    transform = transforms.Compose([
        transforms.Resize((args.resize_watermark, args.resize_watermark)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
   
    if isinstance(watermark_image, str):
        img = Image.open(watermark_image).convert('L')
    else:
        img = watermark_image.convert('L')
   
    img_tensor = transform(img)  # Shape: (1, H, W)
    print("Input tensor shape:", img_tensor.shape, flush=True)
    
    img_tensor = img_tensor.unsqueeze(0).to(device)
   
    with torch.no_grad():
        mu, logvar = encoder(img_tensor)
        print("Mu shape:", mu.shape, flush=True)
        print("Logvar shape:", logvar.shape, flush=True)
        
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    
    latent_sample = mu + eps * std
    print("Latent sample shape:", latent_sample.shape, flush=True)

    for i in tqdm(range(args.start, args.end)):

        
            seed = i + args.gen_seed
            os.makedirs(f"output_trial_{args.resize_watermark}_{args.target_channel}_{TRANSFORMATION}/_{args.run_name}/{i}/", exist_ok=True)
            
            current_prompt = dataset[i][prompt_key]
            print(current_prompt)
            set_random_seed(seed)
            init_latents_no_w = pipe.get_random_latents()
            
            outputs_no_w = pipe(
                    current_prompt,
                    num_images_per_prompt=args.num_images,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    latents=init_latents_no_w,
                )
                
            orig_image_no_w = outputs_no_w.images[0]
                
                
           
            if init_latents_no_w is None:
                    set_random_seed(seed)
                    init_latents_w = pipe.get_random_latents()
            else:
                    init_latents_w = copy.deepcopy(init_latents_no_w)
        
                
            orig_image_no_w.save(f"output_trial_{args.resize_watermark}_{args.target_channel}_{TRANSFORMATION}/_{args.run_name}/{i}/no_w_image.png")
              
            outputs_w = pipe(
                    current_prompt,
                    num_images_per_prompt=args.num_images,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.image_length,
                    width=args.image_length,
                    latents=init_latents_w,
                    process_condition=True,
                    watermark_image = latent_sample,
                    watermark_size = int((args.resize_watermark)/8),
                    target_channel = int(args.target_channel)
                    )
            orig_image_w = outputs_w.images[0]
            orig_image_w.save(f"output_trial_{args.resize_watermark}_{args.target_channel}_{TRANSFORMATION}/_{args.run_name}/{i}/w_image.png")
                
        
              
            orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)
        
              
            img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
            image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
                
            img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
            image_latents_w = pipe.get_image_latents(img_w, sample=False)
            
            extracted_with_watermark = extract_watermark(image_latents_w, image_latents_no_w, args.target_channel, extract_size=(args.resize_watermark//8, args.resize_watermark//8), upscale_factor=1)
            extracted_without_watermark = extract_watermark(image_latents_no_w, image_latents_w, args.target_channel, extract_size=(args.resize_watermark//8, args.resize_watermark//8), upscale_factor=1)
            with torch.no_grad():
                decoded_image_tensor_with_watermark = decoder(extracted_with_watermark.float())
                decoded_image_tensor_without_watermark = decoder(extracted_without_watermark.float())
          
            to_pil = ToPILImage()
            decoded_with_img = to_pil(decoded_image_tensor_with_watermark.squeeze(0))
            decoded_without_img = to_pil(decoded_image_tensor_without_watermark.squeeze(0))

            decoded_with_img.save(f"output_trial_{args.resize_watermark}_{args.target_channel}_{TRANSFORMATION}/_{args.run_name}/{i}/decoded_with.png")
            decoded_without_img.save(f"output_trial_{args.resize_watermark}_{args.target_channel}_{TRANSFORMATION}/_{args.run_name}/{i}/decoded_without.png")

            no_w_metric, w_metric = evaluation_watermark(decoded_image_tensor_without_watermark, decoded_image_tensor_with_watermark,img_tensor, args, i)
       
            if args.reference_model is not None:
                sims = measure_similarity([orig_image_no_w, orig_image_w], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
                w_no_sim = sims[0].item()
                w_sim = sims[1].item()
            else:
                w_no_sim = 0
                w_sim = 0
            results.append({
                'no_w_metric': no_w_metric, 'w_metric': w_metric, 'w_no_sim': w_no_sim, 'w_sim': w_sim,
            })
            no_w_metrics.append(-no_w_metric)
            w_metrics.append(-w_metric)
            if args.with_tracking:
                if (args.reference_model is not None) and (i < args.max_num_log_image):
                    table.add_data(wandb.Image(orig_image_no_w), w_no_sim, wandb.Image(orig_image_w), w_sim, current_prompt, no_w_metric, w_metric)
                else:
                    table.add_data(None, w_no_sim, None, w_sim, current_prompt, no_w_metric, w_metric)
                clip_scores.append(w_no_sim)
                clip_scores_w.append(w_sim)
       

         # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = sk_metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = sk_metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve_{args.run_name}_{args.target_channel}_{args.resize_watermark}')
    plt.legend(loc="lower right")
    plt.grid(True)
    
  
    roc_curve_path = f"output_trial_{args.resize_watermark}_{args.target_channel}_{TRANSFORMATION}/_{args.run_name}/roc_curve.png"
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC curve saved at {roc_curve_path}")
    if args.with_tracking:
             wandb.log({'Table': table})
             wandb.log({'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                        'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                        'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
       
             print(f'clip_score_mean: {mean(clip_scores)}')
             print(f'w_clip_score_mean: {mean(clip_scores_w)}')
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
    output_file = "metrics_output.json"
   
    metrics = {
        "run_name": args.run_name,
        "start": args.start,
        "end": args.end,
        "target_channel": args.target_channel,
        "watermark size" : args.resize_watermark,
        "clip_score_mean": mean(clip_scores),
        "w_clip_score_mean": mean(clip_scores_w),
        "auc": auc,
        "acc": acc,
        "TPR@1%FPR": low,
    }
 
    with open(output_file, "a") as file:
         json.dump(metrics, file)
         file.write("\n") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--resize_watermark', default=256, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=2, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--target_channel', default=2, type=int)
    parser.add_argument('--scale', default=5, type=float)
    parser.add_argument('--output_file', type=str, default="metrics_output.json", help="Path to the output JSON file for metrics.")
    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)
