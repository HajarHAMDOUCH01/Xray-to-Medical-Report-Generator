import torch
import torch.nn as nn
import os
import json
from typing import Optional, Dict, Any, Union
from PIL import Image
import open_clip
from transformers import BioGptForCausalLM, BioGptTokenizer, PretrainedConfig, PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Import the model components
from .q_former import Qformer, BertConfig
from .encoder import BiomedCLIPEncoder

class XrayReportGeneratorConfig(PretrainedConfig):
    """Configuration class for XrayReportGenerator model."""
    
    model_type = "xray_report_generator"
    
    def __init__(
        self,
        biomedclip_config: Dict[str, Any] = None,
        qformer_config: Dict[str, Any] = None,
        biogpt_config: Dict[str, Any] = None,
        projection_layer: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None,
        checkpoint_files: Dict[str, str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Default configurations
        self.biomedclip_config = biomedclip_config or {
            "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "feature_dim": 512,
            "checkpoint_file": "biomedclip_finetuned.pth"
        }
        
        self.qformer_config = qformer_config or {
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "add_cross_attention": True,
            "cross_attention_freq": 1,
            "encoder_width": 512,
            "num_query_tokens": 32,
            "gradient_checkpointing": False,
            "max_position_embeddings": 1024,
            "position_embedding_type": "absolute"
        }
        
        self.biogpt_config = biogpt_config or {
            "model_name": "microsoft/biogpt",
            "checkpoint_file": "biogpt_finetuned.pth"
        }
        
        self.projection_layer = projection_layer or {
            "enabled": True,
            "input_size": 768,
            "output_size": 1024
        }
        
        self.training_config = training_config or {
            "max_seq_length": 256,
            "biomedclip_encoder_width": 512
        }
        
        self.checkpoint_files = checkpoint_files or {
            "final_model": "final_model.pth",
            "biomedclip": "biomedclip_finetuned.pth",
            "biogpt": "biogpt_finetuned.pth"
        }

class XrayReportGenerator(PreTrainedModel):
    """
    X-ray Report Generator combining BiomedCLIP, Q-Former, and BioGPT.
    """
    config_class = XrayReportGeneratorConfig
    
    def __init__(self, config: XrayReportGeneratorConfig):
        super().__init__(config)
        
        self.config = config
        # Don't set self.device - use self.device property from PreTrainedModel
        
        # Initialize BiomedCLIP encoder (without fine-tuned weights initially)
        self.biomedclip_encoder = BiomedCLIPEncoder(
            model_name=config.biomedclip_config["model_name"],
            weights_path=None,  # Fine-tuned weights loaded later in from_pretrained
            device=None  # Let it determine device automatically
        )
        
        # Initialize Q-Former
        qformer_bert_config = BertConfig(**config.qformer_config)
        self.qformer = Qformer(qformer_bert_config)
        
        # Initialize BioGPT (will load fine-tuned weights later in from_pretrained)
        self.tokenizer = BioGptTokenizer.from_pretrained(config.biogpt_config["model_name"])
        self.biogpt_decoder = BioGptForCausalLM.from_pretrained(config.biogpt_config["model_name"])
        
        # Projection layer
        biogpt_hidden_size = self.biogpt_decoder.config.hidden_size
        if config.projection_layer["enabled"] and config.qformer_config["hidden_size"] != biogpt_hidden_size:
            self.qformer_output_to_biogpt_input_projection = nn.Linear(
                config.qformer_config["hidden_size"], 
                biogpt_hidden_size
            )
        else:
            self.qformer_output_to_biogpt_input_projection = None
        
        # Set up tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.warning("Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")
        
        # Move to device after initialization
        # Note: Don't move biomedclip_encoder here, it handles its own device management
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        config: Optional[XrayReportGeneratorConfig] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        """
        Load a pretrained model from a local directory or Hugging Face Hub.
        """
        from transformers.utils import cached_file
        
        # Load config - HuggingFace will handle downloading
        if config is None:
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path, 
                    "config.json",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision
                )
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                config = cls.config_class(**config_dict)
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using default config.")
                config = cls.config_class()
        
        # Initialize model with config (this loads default weights for BioGPT/BiomedCLIP)
        model = cls(config)
        
        # Load checkpoint files from HuggingFace Hub
        checkpoint_files = config.checkpoint_files
        
        # 1. Try to load final model state first
        try:
            final_model_file = cached_file(
                pretrained_model_name_or_path,
                checkpoint_files["final_model"],
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision
            )
            logger.info(f"Loading final model weights from {final_model_file}")
            state_dict = torch.load(final_model_file, map_location='cpu')
            
            # Handle potential DataParallel wrapper
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            
            # Load with strict=False to handle any missing keys gracefully
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys when loading final model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading final model: {unexpected_keys}")
                
            logger.info("Final model weights loaded successfully.")
            return model
            
        except Exception as e:
            logger.warning(f"Could not load final model weights: {e}. Loading individual components.")
        
        # 2. Load individual component weights if final model failed
        # Load BiomedCLIP weights
        try:
            biomedclip_file = cached_file(
                pretrained_model_name_or_path,
                checkpoint_files["biomedclip"],
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision
            )
            logger.info(f"Loading BiomedCLIP weights from {biomedclip_file}")
            biomedclip_state = torch.load(biomedclip_file, map_location='cpu')
            model.biomedclip_encoder.model.load_state_dict(biomedclip_state, strict=False)
            logger.info("BiomedCLIP weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load BiomedCLIP weights: {e}. Using default weights.")
        
        # Load BioGPT weights
        try:
            biogpt_file = cached_file(
                pretrained_model_name_or_path,
                checkpoint_files["biogpt"],
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision
            )
            logger.info(f"Loading BioGPT weights from {biogpt_file}")
            biogpt_state = torch.load(biogpt_file, map_location='cpu')
            model.biogpt_decoder.load_state_dict(biogpt_state, strict=False)
            logger.info("BioGPT weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load BioGPT weights: {e}. Using default weights.")
        
        return model
    
    def _prepare_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Prepare image features with proper shape and device"""
        image_features = image_features.to(self.device)
        
        # Handle different input shapes based on BiomedCLIP encoder output
        if image_features.ndim == 1:
            # Single feature vector (512,) -> (1, 512)
            image_features = image_features.unsqueeze(0)
        elif image_features.ndim == 2:
            pass
        elif image_features.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected image_features shape: {image_features.shape}")
        
        return image_features

    def forward(
        self,
        # args for inference
        image_path: Optional[str] = None, 
        prompt_text: Optional[str] = None,
        max_new_tokens: int = 50,
        num_beams: int = 1,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        # args for training
        image_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training or inference.
        """
        is_training = image_features is not None and input_ids is not None and attention_mask is not None

        # Get image features
        if image_path is not None and not is_training:
            image_features = self.biomedclip_encoder.encode_image(image_path)
        elif image_features is None:
            raise ValueError("Either image_path or image_features must be provided")
        
        image_features = self._prepare_image_features(image_features)
        query_embeddings = self.qformer(image_features)

        if self.qformer_output_to_biogpt_input_projection:
            query_embeddings = self.qformer_output_to_biogpt_input_projection(query_embeddings)
        
        if is_training:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            report_embeddings = self.biogpt_decoder.get_input_embeddings()(input_ids)
            decoder_input_embeddings = torch.cat([query_embeddings, report_embeddings], dim=1)

            query_attention_mask = torch.ones(
                query_embeddings.shape[0],
                query_embeddings.shape[1],
                dtype=torch.long,
                device=query_embeddings.device
            )

            decoder_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)
            labels = input_ids.clone()
            ignored_labels_for_query = torch.full(
                (query_embeddings.shape[0], query_embeddings.shape[1]),
                -100,
                dtype=torch.long,
                device=query_embeddings.device
            )
            decoder_labels = torch.cat([ignored_labels_for_query, labels], dim=1)
            biogpt_decoder_kwargs = {
                "inputs_embeds": decoder_input_embeddings,
                "attention_mask": decoder_attention_mask,
                "labels": decoder_labels,
                "return_dict": True
            }
            outputs = self.biogpt_decoder(**biogpt_decoder_kwargs) 
            return outputs.loss
        else:
            # Inference mode
            input_embeddings = query_embeddings
            input_attention_mask = torch.ones_like(input_embeddings[:, :, 0], dtype=torch.long)
            
            # Add prompt if provided
            if prompt_text:
                prompt_token_ids = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=32,
                    truncation=True
                ).input_ids.to(self.device)
                
                text_embeddings = self.biogpt_decoder.get_input_embeddings()(prompt_token_ids)
                input_embeddings = torch.cat([input_embeddings, text_embeddings], dim=1)
                input_attention_mask = torch.cat([
                    input_attention_mask,
                    torch.ones_like(prompt_token_ids)
                ], dim=1)

            generation_kwargs = {
                "inputs_embeds": input_embeddings,
                "attention_mask": input_attention_mask,
                "max_new_tokens": 256,
                "min_length": 20,
                "num_beams": 5,
                "temperature": 1.0,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.5,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "length_penalty": 2.0,
                "no_repeat_ngram_size": 3
            }

            outputs = self.biogpt_decoder.generate(**generation_kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_report(
        self, 
        image_path: str, 
        prompt_text: Optional[str] = None,
        max_new_tokens: int = 256,
        num_beams: int = 5,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.5,
        **kwargs
    ) -> str:
        """
        Generate a medical report from an X-ray image.
        
        Args:
            image_path (str): Path to the X-ray image
            prompt_text (Optional[str]): Optional prompt text
            max_new_tokens (int): Maximum number of tokens to generate
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for sampling
            top_p (float): Top-p value for nucleus sampling
            do_sample (bool): Whether to use sampling
            repetition_penalty (float): Repetition penalty
            
        Returns:
            str: Generated medical report
        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                image_path=image_path,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                top_p=top_p,
                **kwargs
            )

# Register the model for AutoModel
from transformers import AutoConfig, AutoModel
AutoConfig.register("xray_report_generator", XrayReportGeneratorConfig)
AutoModel.register(XrayReportGeneratorConfig, XrayReportGenerator)

