import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BioGptTokenizer
from transformers.optimization import get_scheduler
from torch.optim import AdamW
import os
import sys
from tqdm.auto import tqdm
import json 
import warnings 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path: 
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to sys.path")

from models.trained_models.BioMedClip.encoder import BiomedCLIPEncoder
from models.trained_models.Q_former.q_former import Qformer, BertConfig
from models.trained_models.biogpt.biogpt_model import XrayReportGenerator
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

# --- Configuration for Training ---
class TrainingConfig:
    def __init__(self):
        self.dataset_dir = "/content/drive/MyDrive/processed_dataset" 
        self.output_dir = "/content/drive/MyDrive/finetuned_report_generator"
        self.max_seq_length = 256
        self.train_batch_size = 4
        self.eval_batch_size = 8 
        self.learning_rate = 5e-5
        self.num_epochs = 3
        self.warmup_steps = 0.1 
        self.gradient_accumulation_steps = 1
        self.biomedclip_encoder_width = 512 

class ReportGenerationDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: BioGptTokenizer, max_seq_length: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = [] 

        # FIXED: Proper special token handling
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = '<s>'
            logger.warning("Set '<s>' as BOS token.")
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '</s>'
            logger.warning("Set '</s>' as EOS token.")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.warning(f"Set pad_token to eos_token: '{self.tokenizer.eos_token}'")
        
        # Load data from JSON files
        logger.info(f"Loading data from directory: {data_dir}")
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'): 
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        entry = json.load(f)
                    self.data.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {filepath}. Skipping.")
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}. Skipping.")
        
        if not self.data:
            raise ValueError(f"No data loaded from {data_dir}. Check directory path and file formats.")
        logger.info(f"Successfully loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_embedding = torch.tensor(item["embedding"], dtype=torch.float32)

        report_text = item["report"]
        if not report_text.startswith(self.tokenizer.bos_token):
            report_text = self.tokenizer.bos_token + report_text
        if not report_text.endswith(self.tokenizer.eos_token):
            report_text = report_text + self.tokenizer.eos_token
            
        tokenized_report = self.tokenizer(
            report_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized_report.input_ids.squeeze(0)
        attention_mask = tokenized_report.attention_mask.squeeze(0)
        
        return {
            "image_embedding": image_embedding,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

# --- Main Training Function ---
def train_model():
    config = TrainingConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {config.output_dir}")

    # Initialize tokenizer
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning("Set pad_token_id to eos_token_id.")
    logger.info("Tokenizer initialized.")

    # Prepare Dataset and DataLoader
    train_dataset = ReportGenerationDataset(config.dataset_dir, tokenizer, config.max_seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    logger.info(f"DataLoader initialized with {len(train_dataloader)} batches.")

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
        encoder_width=config.biomedclip_encoder_width,  # 512
        num_query_tokens=32,
        gradient_checkpointing=False,
        max_position_embeddings=1024,
        position_embedding_type="absolute"
    )
    logger.info("Q-Former BertConfig prepared.")

    logger.info("Initializing XrayReportGenerator model...")
    model = XrayReportGenerator(
        biomedclip_model_name=MODEL_NAMES['biomedclip'],
        biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'],
        qformer_config=qformer_config,  # Pass the config object directly
        biogpt_weights_path=MODEL_WEIGHTS.get('biogpt', None)  # Use get() to handle missing key
    ).to(device)
    logger.info("XrayReportGenerator model instantiated and moved to device.")

    model.biomedclip_encoder.eval()
    for param in model.biomedclip_encoder.parameters():
        param.requires_grad = False
    logger.info("BiomedCLIP encoder frozen.")

    trainable_params = []
    
    trainable_params.extend(list(model.qformer.parameters()))
    logger.info(f"Q-Former parameters: {sum(p.numel() for p in model.qformer.parameters())}")
    
    if model.qformer_output_to_biogpt_input_projection is not None:
        trainable_params.extend(list(model.qformer_output_to_biogpt_input_projection.parameters()))
        logger.info("Projection layer parameters added to training.")
    
    # BioGPT parameters (trainable)
    trainable_params.extend(list(model.biogpt_decoder.parameters()))
    logger.info(f"BioGPT parameters: {sum(p.numel() for p in model.biogpt_decoder.parameters())}")
    
    total_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    # Optimizer
    optimizer = AdamW(trainable_params, lr=config.learning_rate) 
    logger.info("Optimizer initialized with trainable parameters.")

    # Learning Rate Scheduler
    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_steps)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    logger.info(f"LR scheduler: {num_training_steps} total steps, {num_warmup_steps} warmup steps")

    # --- Training Loop ---
    logger.info("Starting training...")
    model.train()
    
    # Set BiomedCLIP to eval even during training because it's already fine tuned 
    model.biomedclip_encoder.eval()
    
    progress_bar = tqdm(range(num_training_steps), desc="Training")
    completed_steps = 0

    for epoch in range(config.num_epochs):
        total_loss_epoch = 0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            image_embedding = batch["image_embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            loss = model(
                image_features=image_embedding,  # Match parameter name in XrayReportGenerator
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Handle gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                progress_bar.update(1)

            total_loss_epoch += loss.item() * config.gradient_accumulation_steps
            epoch_steps += 1

            # Log progress
            if completed_steps % 100 == 0 and (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                logger.info(f"Step {completed_steps}/{num_training_steps}, "
                           f"Epoch {epoch+1}/{config.num_epochs}, "
                           f"Batch {batch_idx+1}/{len(train_dataloader)}, "
                           f"Loss: {current_loss:.4f}")

        # Epoch summary
        avg_loss_epoch = total_loss_epoch / epoch_steps
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} completed. Average Loss: {avg_loss_epoch:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_loss_epoch,
            'config': config.__dict__
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    progress_bar.close()
    logger.info("Training completed!")

    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    # Save tokenizer
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Tokenizer saved to {config.output_dir}")

    config_save_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    logger.info(f"Training config saved: {config_save_path}")

if __name__ == "__main__":
    train_model()