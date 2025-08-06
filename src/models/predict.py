import torch
from PIL import Image
from transformers import BioGptTokenizer
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import your custom modules
from src.models.trained_models.biogpt.biogpt_model import XrayReportGenerator
from src.models.trained_models.Q_former.q_former import BertConfig
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, device):
    """Load the trained model with proper configuration"""
    # Initialize Q-Former config (should match training config)
    qformer_config = BertConfig(
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
        encoder_width=512,  # Must match BiomedCLIP feature dim
        num_query_tokens=32,
        gradient_checkpointing=False,
        max_position_embeddings=1024,
        position_embedding_type="absolute"
    )
    
    # Initialize model with original weights for both BioMedCLIP and BioGPT
    model = XrayReportGenerator(
        biomedclip_model_name=MODEL_NAMES['biomedclip'],
        biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'],
        qformer_config=qformer_config,
        biogpt_weights_path=MODEL_WEIGHTS['biogpt']  # Original BioGPT weights
    ).to(device)
    
    # Now load the fine-tuned weights for the entire model
    if model_path:
        logger.info(f"Loading fine-tuned model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model.eval()
    return model

def generate_report(model, image_path, tokenizer, device, generation_params):
    """Generate radiology report for a given image"""
    try:
        with torch.no_grad():
            generated_report = model(
                image_path=image_path,
                max_new_tokens=generation_params['max_length'],
                num_beams=generation_params['num_beams'],
                do_sample=generation_params['do_sample'],
                top_k=generation_params['top_k'],
                top_p=generation_params['top_p']
            )
        return generated_report
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return None

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate radiology reports from X-ray images")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input X-ray image')
    parser.add_argument('--model_path', type=str, default='final_model.pth', 
                       help='Path to the fine-tuned model weights (default: final_model.pth)')
    parser.add_argument('--output_file', type=str, default=None, 
                       help='Optional file to save the report (default: print to console)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference (cuda/cpu)')
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=256, 
                       help='Maximum length of generated report')
    parser.add_argument('--num_beams', type=int, default=3, 
                       help='Number of beams for beam search')
    parser.add_argument('--do_sample', action='store_true', 
                       help='Use sampling instead of greedy decoding')
    parser.add_argument('--top_k', type=int, default=50, 
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9, 
                       help='Top-p (nucleus) sampling parameter')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    if args.model_path and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    
    # Set up device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer (should be the same one used in training)
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path, device)
    
    # Prepare generation parameters
    generation_params = {
        'max_length': args.max_length,
        'num_beams': args.num_beams,
        'do_sample': args.do_sample,
        'top_k': args.top_k,
        'top_p': args.top_p
    }
    
    # Generate report
    logger.info(f"Generating report for image: {args.image_path}")
    report = generate_report(model, args.image_path, tokenizer, device, generation_params)
    
    if report is None:
        logger.error("Failed to generate report")
        return
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {args.output_file}")
    else:
        print("\nGenerated Report:")
        print("="*50)
        print(report)
        print("="*50)

if __name__ == "__main__":
    main()