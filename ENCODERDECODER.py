import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image
import random
import io
import os

# -------------------------
# VAE Components (Encoder, Decoder, VAE)
# -------------------------
class VAEEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=1):
        super(VAEEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, 32)
        self.act1 = nn.SiLU()
        
        self.down1 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(32, 64)
        self.act2 = nn.SiLU()
        
        self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(32, 128)
        self.act3 = nn.SiLU()
        
        self.down3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.GroupNorm(32, 256)
        self.act4 = nn.SiLU()
        
        self.conv_mu = nn.Conv2d(256, latent_channels, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv2d(256, latent_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.down1(x)))
        x = self.act3(self.norm3(self.down2(x)))
        x = self.act4(self.norm4(self.down3(x)))
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_channels=1, out_channels=1):
        super(VAEDecoder, self).__init__()
        self.conv1 = nn.Conv2d(latent_channels, 256, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, 256)
        self.act1 = nn.SiLU()
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 64),
            nn.SiLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 32),
            nn.SiLU()
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()
    
    def forward(self, z):
        z = self.act1(self.norm1(self.conv1(z)))
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.final_act(self.final_conv(z))
        return z

class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=1, out_channels=1):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(in_channels, latent_channels)
        self.decoder = VAEDecoder(latent_channels, out_channels)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# -------------------------
# VGG Feature Extractor for Perceptual Loss
# -------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=8):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice = nn.Sequential()
        for x in range(layer_index):
            self.slice.add_module(str(x), vgg[x])
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.slice(x)

def prepare_vgg_input(x):
    # Scale from [-1,1] to [0,1]
    x = (x + 1) / 2
    x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
    x = (x - mean) / std
    return x

# -------------------------
# Random Augmentation with Random Parameters
# -------------------------
def apply_random_single_augmentation(image, input_size,
                                     r_degree_range=(0, 75),
                                     jpeg_quality_range=(20, 25),
                                     crop_scale_range=(0.5, 0.75),
                                     crop_ratio_range=(0.5, 0.75),
                                     gaussian_blur_r_range=(2, 4),
                                     gaussian_std_range=(0.05, 0.1),
                                     brightness_range=(0.5, 6)):
    """
    Randomly choose one augmentation (rotate, crop, jpeg, gaussian_blur, brightness) or none.
    For the chosen augmentation, sample parameters from the given ranges.
    Finally, resize the image to input_size.
    Returns the augmented image and the chosen augmentation type.
    """
    aug_choice = random.choice(["rotate", "crop", "jpeg", "gaussian_blur", "brightness", "none"])
    
    if aug_choice == "rotate":
        # Sample an angle from -max to max.
        angle = random.uniform(-r_degree_range[1], r_degree_range[1])
        image = TF.rotate(image, angle)
        
    elif aug_choice == "crop":
        # Randomly sample crop scale and crop ratio.
        crop_scale = random.uniform(crop_scale_range[0], crop_scale_range[1])
        crop_ratio = random.uniform(crop_ratio_range[0], crop_ratio_range[1])
        w, h = image.size
        target_area = crop_scale * w * h
        crop_w = int(round((target_area * crop_ratio) ** 0.5))
        crop_h = int(round((target_area / crop_ratio) ** 0.5))
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        left = random.randint(0, w - crop_w) if w - crop_w > 0 else 0
        top = random.randint(0, h - crop_h) if h - crop_h > 0 else 0
        image = TF.crop(image, top, left, crop_h, crop_w)
        
    elif aug_choice == "jpeg":
        # Randomly choose JPEG quality.
        quality = random.randint(jpeg_quality_range[0], jpeg_quality_range[1])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert('L')
        
    elif aug_choice == "gaussian_blur":
        # Randomly choose blur radius and sigma.
        blur_r = random.randint(gaussian_blur_r_range[0], gaussian_blur_r_range[1])
        kernel_size = 2 * blur_r + 1
        std_val = random.uniform(gaussian_std_range[0], gaussian_std_range[1])
        blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(std_val, std_val))
        image = blur(image)
        
    elif aug_choice == "brightness":
        # Randomly sample brightness factor.
        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        image = TF.adjust_brightness(image, brightness_factor)
    else:
        # "none" means no augmentation.
        pass
    
    image = TF.resize(image, input_size)
    return image, aug_choice

# -------------------------
# Training Setup with Mixed Batch, VGG Loss, Scheduler, and Random Augmentation Parameters
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Desired input size (must be divisible by 8)
    input_size = (128, 128)
    
    # Instead of fixed values, we now pass ranges to the augmentation function.
    r_degree_range = (0, 75)
    jpeg_quality_range = (20, 25)
    crop_scale_range = (0.5, 0.75)
    crop_ratio_range = (0.5, 0.75)
    gaussian_blur_r_range = (2, 4)
    gaussian_std_range = (0.05, 0.1)
    brightness_range = (0.5, 6)
    
    apply_same_augmentation_for_target = True
    
    true_batch_size = 64
    false_batch_size = 64  # Total batch = true + false
    
    # Load the single reference (true) image.
    ref_image_path = '/home/nitesh-mtech/tree-ring-watermark-main/OK.png'
    if not os.path.exists(ref_image_path):
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
    ref_image = Image.open(ref_image_path).convert('L')
    
    # Clean transform (resize and normalize to [-1,1]).
    clean_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    clean_target = clean_transform(ref_image)
    
    # Folder for false images.
    false_images_folder = '/home/nitesh-mtech/tree-ring-watermark-main/coco/test2017/test2017'
    false_image_paths = [os.path.join(false_images_folder, f) for f in os.listdir(false_images_folder)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(false_image_paths) == 0:
        raise FileNotFoundError("No false images found in the specified folder.")
    
    # Fixed false target (noise image normalized to [-1,1]).
    false_target_tensor = torch.randn(1, 1, input_size[0], input_size[1])
    false_target_tensor = (false_target_tensor - false_target_tensor.min()) / (false_target_tensor.max() - false_target_tensor.min())
    false_target_tensor = false_target_tensor * 2 - 1
    
    # Create the VAE model.
    model = VAE(in_channels=1, latent_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Scheduler: StepLR decays LR by 0.95 every 1000 epochs.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    
    # Initialize VGG feature extractor for perceptual loss.
    vgg_extractor = VGGFeatureExtractor(layer_index=8).to(device)
    vgg_extractor.eval()
    vgg_weight = 0.1  # Weight for perceptual loss.
    
    num_epochs = 100000
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)
    
    print("Starting mixed-batch training with VGG perceptual loss and random augmentation parameters...")
    for epoch in range(num_epochs):
        model.train()
        true_inputs = []
        true_targets = []
        # Build true batch.
        for i in range(true_batch_size):
            if random.choice([True, False]):
                aug_img, _ = apply_random_single_augmentation(
                    ref_image, input_size,
                    r_degree_range, jpeg_quality_range,
                    crop_scale_range, crop_ratio_range,
                    gaussian_blur_r_range, gaussian_std_range,
                    brightness_range
                )
                true_inputs.append(transforms.ToTensor()(aug_img))
                if apply_same_augmentation_for_target:
                    true_targets.append(transforms.ToTensor()(aug_img))
                else:
                    true_targets.append(clean_target)
            else:
                img = TF.resize(ref_image, input_size)
                true_inputs.append(transforms.ToTensor()(img))
                true_targets.append(clean_target)
        true_inputs = torch.stack(true_inputs)
        true_targets = torch.stack(true_targets)
        
        false_inputs = []
        # Build false batch.
        for i in range(false_batch_size):
            false_path = random.choice(false_image_paths)
            false_img = Image.open(false_path).convert('L')
            false_img = TF.resize(false_img, input_size)
            false_inputs.append(transforms.ToTensor()(false_img))
        false_inputs = torch.stack(false_inputs)
        
        normalize = transforms.Normalize((0.5,), (0.5,))
        true_inputs = normalize(true_inputs)
        false_inputs = normalize(false_inputs)
        
        batch_inputs = torch.cat([true_inputs, false_inputs], dim=0).to(device)
        batch_true_targets = true_targets.to(device)
        batch_false_targets = false_target_tensor.repeat(false_batch_size, 1, 1, 1).to(device)
        batch_targets = torch.cat([batch_true_targets, batch_false_targets], dim=0)
        
        recon, mu, logvar = model(batch_inputs)
        recon_loss = F.mse_loss(recon, batch_targets)
        
        # Compute perceptual loss using VGG.
        recon_vgg = prepare_vgg_input(recon)
        target_vgg = prepare_vgg_input(batch_targets)
        vgg_features_recon = vgg_extractor(recon_vgg)
        vgg_features_target = vgg_extractor(target_vgg)
        perceptual_loss = F.mse_loss(vgg_features_recon, vgg_features_target)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (true_batch_size + false_batch_size)
        loss = recon_loss + vgg_weight * perceptual_loss + 1e-5 * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Total Loss={loss.item():.4f}, Recon Loss={recon_loss.item():.4f}, "
                  f"Perceptual Loss={perceptual_loss.item():.4f}, KL Loss={kl_loss.item():.4f} | LR: {current_lr:.6f}")
        
        if epoch % 500 == 0:
            save_image(batch_inputs, os.path.join(output_folder, f"input_epoch_{epoch}.png"), normalize=True)
            save_image(recon, os.path.join(output_folder, f"recon_epoch_{epoch}.png"), normalize=True)
            save_image(batch_targets, os.path.join(output_folder, f"target_epoch_{epoch}.png"), normalize=True)
    
    print("Training completed.")
    
    # -------------------------
    # Testing on a Different Image (expected to mimic false target behavior)
    # -------------------------
    test_image_path = '/home/nitesh-mtech/tpimgcuda/images/lena_rgb.png'
    if os.path.exists(test_image_path):
        test_image = Image.open(test_image_path).convert('L')
        test_image = TF.resize(test_image, input_size)
        test_tensor = transforms.ToTensor()(test_image)
        test_tensor = normalize(test_tensor)
        test_tensor = test_tensor.unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            recon_test, _, _ = model(test_tensor)
        print("Test image reconstruction (expected to differ from true target).")
        save_image(test_tensor, os.path.join(output_folder, "test_input.png"), normalize=True)
        save_image(recon_test, os.path.join(output_folder, "test_recon.png"), normalize=True)
    else:
        print("No test image provided; training was performed on the mixed batch.")
    
    model_save_path = os.path.join(output_folder, 'vae_model3.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")
