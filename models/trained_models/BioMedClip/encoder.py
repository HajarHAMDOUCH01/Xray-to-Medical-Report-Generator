import torch
import torch.nn as nn
from PIL import Image
import open_clip
import os

project_path = '/content/report_generator'
os.chdir(project_path)

import sys
if project_path not in sys.path:
    sys.path.append(project_path)
    print(f"Added '{project_path}' to sys.path")

from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

class BiomedCLIPEncoder(nn.Module):  
    def __init__(self, model_name=MODEL_NAMES['biomedclip'],
                 weights_path=MODEL_WEIGHTS['biomedclip'],
                 device=None):  
        super().__init__()  
        
        # FIX: Use provided device or auto-detect
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        
        # FIX: Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = 512
    
    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0)
        
        # FIX: Use self.device instead of detecting from model
        image_input = image_input.to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        return image_features