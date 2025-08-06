import torch
import torch.nn as nn
import os
import sys
import json
import argparse
from PIL import Image
import logging
from typing import Optional, Dict, Any
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to sys.path")

from models.trained_models.BioMedClip.encoder import BiomedCLIPEncoder
from models.trained_models.Q_former.q_former import Qformer, BertConfig
from models.trained_models.biogpt.biogpt_model import XrayReportGenerator
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

class XrayReportPredictor:
    """
    A prediction class for generating X-ray reports from images using the trained model.
    """
    
    def __init__(self, 
                 model_checkpoint_path: str,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the predictor with a trained model checkpoint.
        
        Args:
            model_checkpoint_path: Path to the trained model checkpoint (.pth file)
            config_path: Path to training configuration (optional)
            device: Device to run inference on (optional, auto-detects if None)
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")
        
        self.model_checkpoint_path = model_checkpoint_path
        self.config_path = config_path
        
        # Load configuration if provided
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        
        logger.info("XrayReportPredictor initialized successfully.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration if available."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        # Default configuration matching training setup
        return {
            "biomedclip_encoder_width": 512,
            "max_seq_length": 256
        }
    
    def _create_qformer_config(self) -> BertConfig:
        """Create Q-Former configuration matching training setup."""
        return BertConfig(
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            add_cross_attention=True,
            cross_attention_freq=1,
            encoder_width=self.config.get("biomedclip_encoder_width", 512),
            num_query_tokens=32,
            gradient_checkpointing=False,
            max_position_embeddings=1024,
            position_embedding_type="absolute"
        )
    
    def _load_model(self) -> XrayReportGenerator:
        """Load the trained model from checkpoint."""
        logger.info("Loading model...")
        
        # Create Q-Former config
        qformer_config = self._create_qformer_config()
        
        # Initialize model architecture
        model = XrayReportGenerator(
            biomedclip_model_name=MODEL_NAMES['biomedclip'],
            biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'],
            qformer_config=qformer_config,
            biogpt_weights_path=MODEL_WEIGHTS.get('biogpt', None)
        )
        
        # Load trained weights
        if os.path.exists(self.model_checkpoint_path):
            logger.info(f"Loading model weights from {self.model_checkpoint_path}")
            
            try:
                # Load checkpoint
                checkpoint = torch.load(self.model_checkpoint_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    state_dict = checkpoint
                
                # Load state dict
                model.load_state_dict(state_dict, strict=True)
                logger.info("Model weights loaded successfully.")
                
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                raise
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_checkpoint_path}")
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Ensure BiomedCLIP encoder is in eval mode and frozen
        model.biomedclip_encoder.eval()
        for param in model.biomedclip_encoder.parameters():
            param.requires_grad = False
        
        return model
    
    def predict(self,
                image_path: str,
                prompt_text: Optional[str] = None,
                max_new_tokens: int = 100,
                num_beams: int = 3,
                do_sample: bool = False,
                top_k: Optional[int] = 50,
                top_p: Optional[float] = 0.9,
                temperature: float = 0.7) -> str:
        """
        Generate a report for the given X-ray image.
        
        Args:
            image_path: Path to the X-ray image
            prompt_text: Optional prompt text to condition generation
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            temperature: Temperature for sampling
            
        Returns:
            Generated report as a string
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        logger.info(f"Generating report for image: {image_path}")
        
        try:
            with torch.no_grad():
                # Generate report using the model
                generated_report = self.model(
                    image_path=image_path,
                    prompt_text=prompt_text,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p
                )
                
                logger.info("Report generated successfully.")
                return generated_report
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_batch(self,
                      image_paths: list,
                      prompt_text: Optional[str] = None,
                      **generation_kwargs) -> list:
        """
        Generate reports for multiple images.
        
        Args:
            image_paths: List of paths to X-ray images
            prompt_text: Optional prompt text to condition generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated reports
        """
        reports = []
        
        for image_path in image_paths:
            try:
                report = self.predict(
                    image_path=image_path,
                    prompt_text=prompt_text,
                    **generation_kwargs
                )
                reports.append(report)
                logger.info(f"Generated report for {image_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate report for {image_path}: {e}")
                reports.append(f"Error: {str(e)}")
        
        return reports


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate X-ray reports using trained model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the X-ray image")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to training configuration file (optional)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Optional prompt text to condition generation")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Optional file to save the generated report")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, auto-detects if not specified)")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--num_beams", type=int, default=3,
                       help="Number of beams for beam search")
    parser.add_argument("--do_sample", action="store_true",
                       help="Use sampling instead of beam search")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for sampling")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = XrayReportPredictor(
            model_checkpoint_path=args.model_path,
            config_path=args.config_path,
            device=args.device
        )
        
        # Generate report
        report = predictor.predict(
            image_path=args.image_path,
            prompt_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            top_k=args.top_k if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            temperature=args.temperature
        )
        
        # Print report
        print("\n" + "="*50)
        print("GENERATED X-RAY REPORT")
        print("="*50)
        print(report)
        print("="*50)
        
        # Save to file if specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.output_file}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()