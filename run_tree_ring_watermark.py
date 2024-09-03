import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
import os
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image
from scipy.fftpack import dct, idct
torch.set_default_dtype(torch.float16)
import torch_dct as dct

def apply_dct(image):
    image_tensor = ToTensor()(image).unsqueeze(0)
    dct_image = dct(dct(image_tensor, axis=-1, norm='ortho'), axis=-2, norm='ortho')
    return dct_image

def apply_idct(dct_image):
    idct_image = idct(idct(dct_image, axis=-1, norm='ortho'), axis=-2, norm='ortho')
    return ToPILImage()(idct_image.squeeze())

def get_dct_latents(pipe, dct_image, device):
    return pipe.get_image_latents(dct_image.to(device), sample=False)

def main(args):
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)
    table = None
    os.environ['REQUESTS_CA_BUNDLE'] = 'C:\\Users\\nites\\Desktop\\Sophos Certificate\\SecurityAppliance_SSL_CA.pem'
    
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    # gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []
    p_values_no_w = []
    p_values_w = []
    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### Generation without watermarking
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
        save_image_to_unique_folder(orig_image_no_w, base_dir, i,"orig_image_no_w")
        
        
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w_original = copy.deepcopy(init_latents_no_w)
        # latents_no_w = get_dct_latents(pipe, dct_image_no_w, device)
        
        # Inject watermark
        # watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        imageforwatermark = readImage("th (1).jpeg",args,i) #grayscale image(if not, will be converted in the function)
        init_latents_w,watermark_size = inject_watermark(init_latents_w_original,imageforwatermark, args,pipe,device,i)
        # save_image_to_unique_folder(init_latents_w, base_dir, i,"init_latents_w_watermarked")
        # Reverse latent space to get watermarked DCT image
        # print(init_latents_w.dtype)
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w.to(torch.float16),
            
        )
        orig_image_w = outputs_w.images[0]
        save_image_to_unique_folder(orig_image_w, base_dir, i,"orig_image_w")

        ### Test watermark
        # Distortion
        orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)
        save_image_to_unique_folder(orig_image_w_auged, base_dir, i,"orig_image_w_auged")
        # Reverse img without watermarking
        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        # save_image_to_unique_folder(img_no_w, base_dir, i,"img_no_w")
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
        # save_image_to_unique_folder(image_latents_no_w, base_dir, i,"image_latents_no_w")

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # Reverse img with watermarking
        img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        # save_image_to_unique_folder(img_w, base_dir, i,"img_w")
        image_latents_w = pipe.get_image_latents(img_w, sample=False)
        # save_image_to_unique_folder(image_latents_w, base_dir, i,"image_latents_w")

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # Evaluation
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w,watermark_size,init_latents_w_original, args)
        # Compute p-values
        p_no_w, p_w = get_p_value(reversed_latents_no_w, reversed_latents_w, imageforwatermark, watermark_size, device,args,pipe,i)
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
        p_values_no_w.append(p_no_w)
        p_values_w.append(p_w)
        
        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                # Log images when using a reference model
                table.add_data(wandb.Image(orig_image_no_w), w_no_sim, wandb.Image(orig_image_w), w_sim, current_prompt, no_w_metric, w_metric)
            else:
                table.add_data(None, w_no_sim, None, w_sim, current_prompt, no_w_metric, w_metric)

            clip_scores.append(w_no_sim)
            clip_scores_w.append(w_sim)
            
            
        # extracted_watermark_latents = extract_watermark(reversed_latents_w, init_latents_w_original, watermark_size, pipe, device,args,i)
        # save_image_to_unique_folder(extracted_watermark_latents, base_dir, i,"extracted_watermark_latents")
        # extracted_watermark_image = latents_to_imgs(pipe, extracted_watermark_latents)
        # save_image_to_unique_folder(extracted_watermark_image[0], base_dir, i, "extracted_watermark")

    # ROC for metrics
    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr < 0.01)[0][-1]]
    max_threshold = thresholds[np.argmax(1 - (fpr + (1 - tpr)) / 2)]
    average_threshold = np.mean(thresholds)
    min_threshold = np.min(thresholds)
    max_threshold_val = np.max(thresholds)
    thresholds_at_1_fpr = thresholds[np.where(fpr < 0.01)[0]]
    
    print("\n\n\n")
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
    print(f'Best Threshold for classification: {max_threshold}')
    print(f'Average Threshold: {average_threshold}')
    print(f'Minimum Threshold: {min_threshold}')
    print(f'Maximum Threshold: {max_threshold_val}')
    print(f'Thresholds at FPR < 1%: {thresholds_at_1_fpr}')
    saveRocCurve(fpr, tpr,base_dir,i)
    print("\n\n\n")
    
    plot_and_save_pw_and_pnow(p_values_no_w, p_values_w, base_dir,i)
    
    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                   'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                   'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
    
        print(f'clip_score_mean: {mean(clip_scores)}')
        print(f'w_clip_score_mean: {mean(clip_scores_w)}')
    
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='train')
    parser.add_argument('--dataset', default='coco')
    # parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--output_dir', default='output_images', type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--image_length', default=256, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex') #seed used for direct injection in spatial doman.. complex in freq domain
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