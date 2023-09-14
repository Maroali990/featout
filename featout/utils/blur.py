from featout.utils.gaussian_smoothing import GaussianSmoothing
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import torch


def blur_around_max(img, max_coordinates, patch_radius=4, kernel_size=5):
    """
    Blur a quadratic patch around max_coordinates with a Gaussian filter
    """
    max_x, max_y = max_coordinates
    modified_input = img.clone()

    # check whether it is in the image bounds
    x_start = max([max_x - patch_radius, 0])
    x_end = min([max_x + patch_radius + 1, modified_input.shape[2]])
    y_start = max([max_y - patch_radius, 0])
    y_end = min([max_y + patch_radius + 1, modified_input.shape[3]])

    patch = modified_input[:, :, x_start:x_end, y_start:y_end]

    # smooth only the patch (padded)
    smoothing = GaussianSmoothing(img.size()[1], kernel_size, sigma=1)
    patch_pad = F.pad(patch, (2, 2, 2, 2), mode='reflect')
    smoothed_patch = smoothing(patch_pad)

    modified_input[:, :, x_start:x_end, y_start:y_end] = smoothed_patch
    return modified_input


def zero_out(img, max_coordinates, patch_radius=4):
    """
    Zero out a quadratic patch around max_coordinates with a Gaussian filter
    """
    max_x, max_y = max_coordinates
    modified_input = img.clone()
    modified_input[:, :, max_x - patch_radius:max_x + patch_radius + 1,
                   max_y - patch_radius:max_y + patch_radius + 1] = 0
    return modified_input

def inject_noise(img, max_coordinates, patch_radius=4):
    max_x, max_y = max_coordinates
    x_start = max(max_x - patch_radius, 0)
    x_end = min(max_x + patch_radius + 1, img.shape[2])
    y_start = max(max_y - patch_radius, 0)
    y_end = min(max_y + patch_radius + 1, img.shape[3])

    patch = img[:, :, x_start:x_end, y_start:y_end]

    noise = torch.randn_like(patch) * 0.1  # Tune the noise level as needed
    modified_patch = patch + noise
    img[:, :, x_start:x_end, y_start:y_end].copy_(modified_patch)

    return img


def adaptive_blur(img, max_coordinates, attention_score, patch_radius=4):
    max_x, max_y = max_coordinates
    kernel_size = int(2 * attention_score + 1)  # Sample way to relate attention to kernel size
    gaussian_blur = GaussianBlur(kernel_size, sigma=1.0)

    x_start = max(max_x - patch_radius, 0)
    x_end = min(max_x + patch_radius + 1, img.shape[2])
    y_start = max(max_y - patch_radius, 0)
    y_end = min(max_y + patch_radius + 1, img.shape[3])

    patch = img[:, :, x_start:x_end, y_start:y_end]
    blurred_patch = gaussian_blur(patch)
    
    img[:, :, x_start:x_end, y_start:y_end] = blurred_patch
    return img

def invert_feature(img, max_coordinates, patch_radius=4):
    max_x, max_y = max_coordinates

    x_start = max(max_x - patch_radius, 0)
    x_end = min(max_x + patch_radius + 1, img.shape[2])
    y_start = max(max_y - patch_radius, 0)
    y_end = min(max_y + patch_radius + 1, img.shape[3])

    img[:, :, x_start:x_end, y_start:y_end] *= -1.0
    return img

def texture_shuffle(img, max_coordinates, patch_radius=4):
    max_x, max_y = max_coordinates

    x_start = max(max_x - patch_radius, 0)
    x_end = min(max_x + patch_radius + 1, img.shape[2])
    y_start = max(max_y - patch_radius, 0)
    y_end = min(max_y + patch_radius + 1, img.shape[3])

    patch = img[:, :, x_start:x_end, y_start:y_end]
    shuffled_patch = patch[:, :, torch.randperm(patch.shape[2]), :]
    shuffled_patch = shuffled_patch[:, :, :, torch.randperm(shuffled_patch.shape[3])]
    
    img[:, :, x_start:x_end, y_start:y_end] = shuffled_patch
    return img

def pixelate(img, max_coordinates, patch_radius=4, factor=2):
    max_x, max_y = max_coordinates

    x_start = max(max_x - patch_radius, 0)
    x_end = min(max_x + patch_radius + 1, img.shape[2])
    y_start = max(max_y - patch_radius, 0)
    y_end = min(max_y + patch_radius + 1, img.shape[3])

    patch = img[:, :, x_start:x_end, y_start:y_end]
    downsampled_patch = F.interpolate(patch, scale_factor=1/factor, mode='bilinear')
    upsampled_patch = F.interpolate(downsampled_patch, size=patch.shape[-2:], mode='bilinear')
    
    img[:, :, x_start:x_end, y_start:y_end] = upsampled_patch
    return img

def invert_color(img, max_coordinates, patch_radius=4):
    max_x, max_y = max_coordinates

    x_start = max(max_x - patch_radius, 0)
    x_end = min(max_x + patch_radius + 1, img.shape[2])
    y_start = max(max_y - patch_radius, 0)
    y_end = min(max_y + patch_radius + 1, img.shape[3])

    img[:, :, x_start:x_end, y_start:y_end] = 1 - img[:, :, x_start:x_end, y_start:y_end]
    return img

def add_noise(img, max_coordinates, patch_radius=4, noise_factor=0.2):
    max_x, max_y = max_coordinates

    x_start = max(max_x - patch_radius, 0)
    x_end = min(max_x + patch_radius + 1, img.shape[2])
    y_start = max(max_y - patch_radius, 0)
    y_end = min(max_y + patch_radius + 1, img.shape[3])

    patch = img[:, :, x_start:x_end, y_start:y_end]
    noise = torch.randn_like(patch) * noise_factor
    
    img[:, :, x_start:x_end, y_start:y_end] = patch + noise
    return img

