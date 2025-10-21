import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import argparse
import sys
import os
from torchvision import transforms

def load_model(model_name='dinov3_vits16', repo_dir=None):
    """
    Load DINOv3 model from local repository clone.
    
    Available models:
    - dinov3_vits16, dinov3_vits16plus (Small)
    - dinov3_vitb16 (Base)
    - dinov3_vitl16 (Large)
    - dinov3_vith16plus (Huge+)
    - dinov3_vit7b16 (Giant, 7B params)
    - dinov3_convnext_tiny, dinov3_convnext_small, dinov3_convnext_base, dinov3_convnext_large
    
    Args:
        model_name: Name of the model variant
        repo_dir: Path to local DINOv3 repository clone. If None, will clone automatically.
    """
    if repo_dir is None:
        # Clone the repository if not provided
        import subprocess
        repo_dir = os.path.expanduser('~/.cache/dinov3')
        if not os.path.exists(repo_dir):
            print(f"Cloning DINOv3 repository to {repo_dir}...")
            subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/dinov3.git', repo_dir], check=True)
    
    # Download weights automatically
    weights_url = f'https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/?download={model_name}'
    
    print(f"Loading model from local repository: {repo_dir}")
    model = torch.hub.load(repo_dir, model_name, source='local', pretrained=True, trust_repo=True)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, device

def make_transform(resize_size=224):
    """Create preprocessing transform for DINOv3"""
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])

def preprocess_image(image_path, target_size=512):
    """
    Load and preprocess an image for DINOv3 while maintaining aspect ratio.
    
    Args:
        image_path: Path to the image
        target_size: Target size for the shorter edge (default: 512, must be divisible by 16)
    """
    img = Image.open(image_path).convert('RGB')
    
    # Maintain aspect ratio - resize so shorter edge matches target_size
    # DINOv3 uses patch size of 16, so dimensions should be divisible by 16
    w, h = img.size
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    
    # Make dimensions divisible by 16 (patch size)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Apply transforms
    img_tensor = transforms.ToTensor()(img_resized)
    img_tensor = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )(img_tensor)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, img_resized

def extract_features(model, img_tensor, device):
    """
    Extract features from DINOv3 model.
    Returns patch features and CLS token feature.
    """
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # Get output from model
        output = model(img_tensor)
        
        # DINOv3 returns a dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
        if isinstance(output, dict):
            cls_features = output['x_norm_clstoken'].cpu().numpy()
            patch_features = output['x_norm_patchtokens'].cpu().numpy()
        else:
            # Fallback: treat output as concatenated tokens
            # First token is CLS, rest are patches
            cls_features = output[:, 0:1, :].cpu().numpy()
            patch_features = output[:, 1:, :].cpu().numpy()
    
    return patch_features, cls_features

def save_features(patch_features, cls_features, output_path='dinov3_features.pkl'):
    """
    Serialize features to disk.
    """
    features_dict = {
        'patch_features': patch_features,
        'cls_features': cls_features
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"Features saved to {output_path}")
    print(f"Patch features shape: {patch_features.shape}")
    print(f"CLS features shape: {cls_features.shape}")
    
    return output_path

def visualize_pca(patch_features, original_img, n_components=3, alpha=0.5):
    """
    Run PCA on patch features and visualize as RGB image.
    
    Args:
        patch_features: Patch features from DINOv3
        original_img: Original PIL Image
        n_components: Number of PCA components (default: 3 for RGB)
        alpha: Transparency for overlay (0=only original, 1=only PCA, default: 0.5)
    """
    # Reshape patch features: [B, N_patches, D] -> [N_patches, D]
    features_2d = patch_features[0]  # Remove batch dimension
    
    # Run PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_2d)
    
    # Calculate grid size based on original image dimensions
    # DINOv3 uses patch size of 16
    w, h = original_img.size
    grid_h = h // 16
    grid_w = w // 16
    
    # Reshape to spatial grid
    pca_features_spatial = pca_features.reshape(grid_h, grid_w, n_components)
    
    # Normalize each component to [0, 1] for visualization
    for i in range(n_components):
        pca_features_spatial[:, :, i] = (pca_features_spatial[:, :, i] - 
                                         pca_features_spatial[:, :, i].min()) / \
                                        (pca_features_spatial[:, :, i].max() - 
                                         pca_features_spatial[:, :, i].min())
    
    # Resize PCA features to match original image size
    from scipy.ndimage import zoom
    original_array = np.array(original_img).astype(np.float32) / 255.0
    
    # Calculate zoom factors for each dimension
    zoom_factors = [h / grid_h, w / grid_w, 1]
    pca_resized = zoom(pca_features_spatial, zoom_factors, order=1)
    
    # Ensure the resized PCA matches the original image dimensions exactly
    pca_resized = pca_resized[:h, :w, :]
    
    # Create overlay by blending original image with PCA visualization
    overlay = alpha * pca_resized + (1 - alpha) * original_array
    overlay = np.clip(overlay, 0, 1)
    
    # Create visualization with three panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # PCA visualization only
    axes[1].imshow(pca_resized)
    axes[1].set_title(f'PCA Features (First {n_components} Components)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (α={alpha})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('dinov3_pca_visualization.png', dpi=150, bbox_inches='tight')
    print("PCA visualization saved to dinov3_pca_visualization.png")
    plt.show()
    
    # Print explained variance
    print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract features from images using DINOv3')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, default='dinov3_features.pkl', 
                        help='Output path for features (default: dinov3_features.pkl)')
    parser.add_argument('--model', type=str, default='dinov3_vits16',
                        choices=['dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16', 
                                'dinov3_vitl16', 'dinov3_vith16plus', 'dinov3_vit7b16',
                                'dinov3_convnext_tiny', 'dinov3_convnext_small',
                                'dinov3_convnext_base', 'dinov3_convnext_large'],
                        help='DINOv3 model variant (default: dinov3_vits16)')
    parser.add_argument('--repo-dir', type=str, default=None,
                        help='Path to local DINOv3 repository clone (will auto-clone if not provided)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency for overlay visualization (0=original only, 1=PCA only, default: 0.5)')
    parser.add_argument('--size', type=int, default=512,
                        help='Target size for shorter edge, must be divisible by 16 (default: 512)')
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Validate size parameter
    if args.size % 16 != 0:
        print(f"Warning: Size {args.size} is not divisible by 16. Adjusting to {(args.size // 16) * 16}")
        args.size = (args.size // 16) * 16
    
    print(f"Using model: {args.model}")
    print("Loading DINOv3 model...")
    model, device = load_model(args.model, args.repo_dir)
    
    print("Loading and preprocessing image...")
    img_tensor, original_img = preprocess_image(args.image_path, target_size=args.size)
    
    print("Extracting features from DINOv3...")
    patch_features, cls_features = extract_features(model, img_tensor, device)
    
    print("\nSaving features to disk...")
    save_features(patch_features, cls_features, args.output)
    
    print("\nRunning PCA and creating visualization...")
    visualize_pca(patch_features, original_img, alpha=args.alpha)
    
    print("\nDone! Features are saved and PCA visualization is ready.")