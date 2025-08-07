import torch
import torch.nn as nn
from PIL import Image
import open_clip
import os

class BiomedCLIPEncoder(nn.Module):  
    def __init__(self, 
                 model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 weights_path=None,
                 device=None):  
        super().__init__()  
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model with default BiomedCLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        
        # Load custom weights if provided
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"BiomedCLIP custom weights loaded from {weights_path}")
            except Exception as e:
                print(f"Error loading custom weights: {e}. Using default BiomedCLIP weights.")
        else:
            print("Using default BiomedCLIP weights.")

        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Feature dimension for BiomedCLIP
        self.feature_dim = 512
    
    def encode_image(self, image_path):
        """Encode a single image from file path."""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            # Handle PIL Image directly
            image = image_path.convert("RGB")
            
        image_input = self.preprocess(image).unsqueeze(0)
        image_input = image_input.to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        return image_features
    
    def encode_batch(self, image_paths):
        """Encode a batch of images."""
        batch_images = []
        for img_path in image_paths:
            if isinstance(img_path, str):
                image = Image.open(img_path).convert("RGB")
            else:
                image = img_path.convert("RGB")
            batch_images.append(self.preprocess(image))
        
        batch_tensor = torch.stack(batch_images).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(batch_tensor)
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        return image_features

