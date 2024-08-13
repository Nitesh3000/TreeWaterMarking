import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
torch.set_default_dtype(torch.float16)
# import datasets
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
import os
from pytorch_wavelets import DTCWTForward, DTCWTInverse

def main(args):
    
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)
    table = None
    os.environ['REQUESTS_CA_BUNDLE'] = 'C:\\Users\\nites\\Desktop\\Sophos Certificate\\SecurityAppliance_SSL_CA.pem'
    print("Starting the process with the following arguments:")
    print(args)
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    dtcwt_forward = DTCWTForward(J=5, biort='near_sym_b', qshift='qshift_b').to(device)
    dtcwt_inverse = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').to(device)
    # scheduler is responsible for guiding the diffusion process.
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    
    # pipe is the main pipeline that integrates the model and scheduler to generate images from text or latent inputs.
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant='fp16',
        safety_checker = None
        )
    pipe = pipe.to(device)
    print(f"Model ID: {args.model_id}")
    
    total_params = (
        sum(p.numel() for p in pipe.unet.parameters()) +
        sum(p.numel() for p in pipe.vae.parameters()) +
        sum(p.numel() for p in pipe.text_encoder.parameters())
    )
    print(f"Total number of parameters in the model: {total_params}")

    print(f"Scheduler: {scheduler}")
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)
        ref_model_params = sum(p.numel() for p in ref_model.parameters())
        print(f"Reference model ID: {args.reference_model}")
        print(f"Reference model parameters: {ref_model_params}")
    dataset, prompt_key = get_dataset(args)
    print(f"Dataset loaded. Number of images in the dataset: {len(dataset)}")
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []
    
    for i in tqdm(range(args.start, args.end)):
        img_dir = os.path.join(base_dir, f"image_{i}")
        os.makedirs(img_dir, exist_ok=True)

        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        print(f"Processing image {i+1}/{args.end} with prompt: {current_prompt}")

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
        orig_image_no_w.save(os.path.join(img_dir, 'no_w_image.png'))
        
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)
        image_list = latents_to_imgs(pipe, init_latents_w)

        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        # grayscale_watermark_Yl, grayscale_watermark_Yh = get_grayscale_watermark(args.watermark_image_path, dtcwt_forward, device)

        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args, i,device)
        # init_latents_w = inject_watermark(init_latents_w, grayscale_watermark_Yl, grayscale_watermark_Yh, args, i, device)
        # image_list = latents_to_imgs(pipe, init_latents_w)
        
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w = outputs_w.images[0]
        orig_image_w.save(os.path.join(img_dir, 'w_image.png'))

        orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)
        orig_image_no_w_auged.save(os.path.join(img_dir, 'distorted_image_no_w.png'))
        orig_image_w_auged.save(os.path.join(img_dir, 'distorted_image_w.png'))

        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args,device)
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

    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    plt.plot(fpr,tpr) 
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC curve')
    plt.show()
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                   'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                   'auc': auc, 'acc':acc, 'TPR@1%FPR': low})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='train')
    parser.add_argument('--dataset', default='coco')
    # parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--output_dir', default='output_images/waveletHighestAndLowestRingsBlurredCOCO1J=3/trial', type=str)
    parser.add_argument('--start', default=60, type=int)
    parser.add_argument('--end', default=70, type=int)
    parser.add_argument('--image_length', default=512, type=int)
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
    parser.add_argument('--watermark_image_path', default="C:\\Users\\nites\\Downloads\\tree-ring-watermark-main (1)\\tree-ring-watermark-main\\Fresh_Apple_PNG_Clip_Art_Image.png", type=str)
    
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
