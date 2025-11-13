"""
CLIP-based Environment Encoder for Welding ROI
Uses Vision-Language model to extract semantic features from ROI
"""

import torch
import clip
from PIL import Image
import numpy as np
import cv2


class CLIPEnvironmentEncoder:
    """
    CLIP-based environment encoder
    Extracts semantic features from welding ROI
    """
    
    def __init__(self, device=None):
        """
        Initialize CLIP model
        
        Args:
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading CLIP model on {self.device}...")
        
        # Load CLIP ViT-B/32 model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Welding ROI-specific prompts (6D)
        self.prompts = [
            "a clear welding ROI with good visibility",
            "a welding ROI with heavy dark shadows",
            "a welding ROI with metal debris and particles",
            "a welding ROI with bright specular reflections",
            "a welding ROI with weld beads obstructing the line",
            "a welding ROI with complex texture and noise"
        ]
        
        # Pre-compute text embeddings
        print("Pre-computing text embeddings...")
        text_tokens = clip.tokenize(self.prompts).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        print(f"CLIP encoder ready! ({len(self.prompts)} semantic dimensions)")
    
    def encode_roi(self, roi_image):
        """
        Encode ROI image to 6D semantic vector
        
        Args:
            roi_image: ROI crop (BGR numpy array)
        
        Returns:
            features: (6,) numpy array of semantic similarities
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_image = Image.fromarray(rgb_image)
        
        # Preprocess and encode
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities with prompts
        similarities = (image_features @ self.text_features.T).squeeze(0)
        
        return similarities.cpu().numpy()  # (6,)
    
    def encode_with_bbox(self, full_image, bbox):
        """
        Extract ROI from full image and encode
        
        Args:
            full_image: Full image (BGR numpy array)
            bbox: (x1, y1, x2, y2) bounding box coordinates
        
        Returns:
            features: (6,) numpy array
        """
        x1, y1, x2, y2 = bbox
        roi_crop = full_image[y1:y2, x1:x2]
        
        if roi_crop.size == 0:
            # Empty ROI, return zeros
            return np.zeros(6, dtype=np.float32)
        
        return self.encode_roi(roi_crop)
    
    def get_feature_names(self):
        """
        Get semantic feature names
        
        Returns:
            list of feature names
        """
        return [
            'clip_clear',
            'clip_shadow',
            'clip_debris',
            'clip_reflection',
            'clip_beads',
            'clip_noise'
        ]


def test_clip_encoder():
    """Test CLIP encoder with sample image"""
    import sys
    
    print("="*60)
    print("CLIP Environment Encoder Test")
    print("="*60)
    
    # Initialize
    encoder = CLIPEnvironmentEncoder()
    
    # Test with dummy image
    dummy_roi = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    
    print("\nEncoding test ROI...")
    features = encoder.encode_roi(dummy_roi)
    
    print(f"\nExtracted features (6D):")
    for name, value in zip(encoder.get_feature_names(), features):
        print(f"  {name:<20}: {value:.4f}")
    
    print("\n✅ CLIP encoder working!")


if __name__ == "__main__":
    test_clip_encoder()
