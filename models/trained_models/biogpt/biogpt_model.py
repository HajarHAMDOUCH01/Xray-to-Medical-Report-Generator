import torch
import torch.nn as nn
from typing import Optional
from transformers import BioGptForCausalLM, BioGptTokenizer
from models.trained_models.BioMedClip.encoder import BiomedCLIPEncoder
from ..Q_former.q_former import Qformer

class XrayReportGenerator(nn.Module):
    def __init__(
        self, 
        biomedclip_model_name, 
        biomedclip_weights_path, 
        qformer_config, 
        biogpt_weights_path: Optional[str] = None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.biomedclip_encoder = BiomedCLIPEncoder(
            model_name = biomedclip_model_name,
            weights_path=biomedclip_weights_path
        )

        assert qformer_config.encoder_width == self.biomedclip_encoder.feature_dim, "Q-Former encoder_width must match BiomedCLIP feature_dim"
        self.qformer = Qformer(qformer_config)

        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.biogpt_decoder = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

        if biogpt_weights_path:
            print(f"Loading fine-tuned BioGpt weights from: {biogpt_weights_path}")
            try:
                state_dict = torch.load(biogpt_weights_path, map_location='cpu')
                self.biogpt_decoder.load_state_dict(state_dict)
                print("Fine-tuned BioGPT weights loaded successfully.")
            except Exception as e:
                print(f"Error loading BioGPT weights: {e}")
                print("Using default pre-trained BioGPT.")
        else:
            print("No fine-tuned BioGPT weights file provided, using default pre-trained BioGPT.")
        
        ################################################
        biogpt_hidden_size = self.biogpt_decoder.config.hidden_size
        if qformer_config.hidden_size != biogpt_hidden_size:
            self.qformer_output_to_biogpt_input_projection = nn.Linear(
                qformer_config.hidden_size, biogpt_hidden_size
            )
        
        else:
            self.qformer_output_to_biogpt_input_projection = None
        self.eos_token_id = self.tokenizer.eos_token_id

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            import warnings
            warnings.warn("Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")
        self.to(self.device)
        
        if hasattr(self.biomedclip_encoder, 'model'):
            self.biomedclip_encoder.model = self.biomedclip_encoder.model.to(self.device)
    
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
    is_training = image_features is not None and input_ids is not None and attention_mask is not None

    #Get image features
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
                device = query_embeddings.device
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
                "inputs_embeds":decoder_input_embeddings,
                "attention_mask": decoder_attention_mask,
                "labels": decoder_labels,
                "return_dict": True
            }
            outputs = self.biogpt_decoder(**biogpt_decoder_kwargs) 
            return outputs.loss
        else:
            input_embeddings_list = [query_embeddings]
            input_attention_mask_list = [torch.ones(
                query_embeddings.shape[0], 
                query_embeddings.shape[1], 
                dtype=torch.long, 
                device=query_embeddings.device
            )]
            if prompt_text:
                prompt_token_ids = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids
                prompt_token_ids = prompt_token_ids.to(self.device)
                text_embeddings = self.biogpt_decoder.get_input_embeddings()(prompt_token_ids)
                input_embeddings_list.append(text_embeddings)
                input_attention_mask_list.append(torch.ones(
                    text_embeddings.shape[0], 
                    text_embeddings.shape[1], 
                    dtype=torch.long, 
                    device=text_embeddings.device
                ))
                input_embeddings = torch.cat(input_embeddings_list, dim=1)
                input_attention_mask = torch.cat(input_attention_mask_list, dim=1)

            # Use the generation parameters passed to the function
            generation_kwargs = {
                "inputs_embeds": input_embeddings,
                "attention_mask": input_attention_mask,
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            
            # Add optional parameters if provided
            if do_sample:
                generation_kwargs.update({
                    "temperature": 0.7,
                    "top_k": top_k if top_k is not None else 50,
                    "top_p": top_p if top_p is not None else 0.9,
                })
            
            generated_output = self.biogpt_decoder.generate(**generation_kwargs)

            # Decode the generated tokens (skip the input tokens)
            input_length = input_embeddings.shape[1]
            generated_tokens = generated_output[0, input_length:] if generated_output.ndim == 2 else generated_output[input_length:]
            generated_report = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return generated_report
