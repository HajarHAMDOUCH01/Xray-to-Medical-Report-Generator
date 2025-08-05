import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import logging
import math
import warnings
from typing import Optional, Tuple, Dict, Any
from transformers.activations import ACT2FN

logger = logging.getLogger(__name__) 

class ModelOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # Enable both dictionary and attribute access
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return self.kwargs[key]
    
    def __setitem__(self, key, value):
        self.kwargs[key] = value
        setattr(self, key, value)
    
    def __contains__(self, key):
        return key in self.kwargs
    
    def __repr__(self):
        return str(self.kwargs)

class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    def __init__(self, last_hidden_state: torch.Tensor, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, hidden_states: Optional[Tuple[torch.Tensor]] = None, attentions: Optional[Tuple[torch.Tensor]] = None, cross_attentions: Optional[Tuple[torch.Tensor]] = None):
        super().__init__(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
        )

def apply_chunking_to_forward(
    forward_fn, chunk_size, seq_len_dim, *input_tensors
):
    """
    Helper function to apply a forward function to chunks of input tensors.
    Fixed to handle variable chunk sizes correctly.
    """
    if chunk_size <= 0:
        return forward_fn(*input_tensors)
    
    seq_len = input_tensors[0].shape[seq_len_dim]
    chunks = []
    
    for i in range(0, seq_len, chunk_size):
        # Calculate actual chunk size for this iteration
        actual_chunk_size = min(chunk_size, seq_len - i)
        chunk_inputs = [t.narrow(seq_len_dim, i, actual_chunk_size) for t in input_tensors]
        chunks.append(forward_fn(*chunk_inputs))
    
    return torch.cat(chunks, dim=seq_len_dim)

class BertConfig:
    """
    Configuration for the Q-Former's internal BERT-like layers.
    """
    def __init__(self,
                  hidden_size: int = 768,
                  num_hidden_layers: int = 6,
                  num_attention_heads: int = 12,
                  intermediate_size: int = 3072,
                  hidden_act: str = "gelu",
                  hidden_dropout_prob: float = 0.1,
                  attention_probs_dropout_prob: float = 0.1,
                  initializer_range: float = 0.02,
                  layer_norm_eps: float = 1e-12,
                  add_cross_attention: bool = True,
                  cross_attention_freq: int = 1,
                  encoder_width: int = 512,
                  num_query_tokens: int = 32,
                  gradient_checkpointing: bool = False,
                  max_position_embeddings: int = 1024,  # Increased for flexibility
                  position_embedding_type: str = "absolute"):
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.add_cross_attention = add_cross_attention
        self.cross_attention_freq = cross_attention_freq
        self.encoder_width = encoder_width
        self.num_query_tokens = num_query_tokens
        self.gradient_checkpointing = gradient_checkpointing
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        
        # Optional parameters (may not be used in Q-Former)
        self.vocab_size = 30522
        self.pad_token_id = 0

class BertSelfAttention(nn.Module):
    """
    Self-attention mechanism for BERT-like models.
    """
    def __init__(self, config: BertConfig, is_cross_attention: bool):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        if (self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transposes a tensor for multi-head attention scores.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        head_mask: Optional[torch.Tensor] = None, 
        encoder_hidden_states: Optional[torch.Tensor] = None, 
        encoder_attention_mask: Optional[torch.Tensor] = None, 
        past_key_value: Optional[Tuple[torch.Tensor]] = None, 
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for self-attention or cross-attention.
        """
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Add relative position embeddings if configured
        if (self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query"):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            
            # Clamp distance to valid range
            distance = torch.clamp(distance, -self.max_position_embeddings + 1, self.max_position_embeddings - 1)
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs_dropped = self.dropout(attention_probs)

        # Apply head mask if provided
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        # Compute context layer
        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)

        return outputs

class BertSelfOutput(nn.Module):
    """
    Output layer for self-attention.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    """
    Complete attention block combining self-attention and output layers.
    """
    def __init__(self, config: BertConfig, is_cross_attention: bool = False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class BertIntermediate(nn.Module):
    """
    Intermediate (feed-forward) layer.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    """
    Output layer for feed-forward network.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    """
    A single Transformer layer with optional cross-attention.
    """
    def __init__(self, config: BertConfig, layer_num: int):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = getattr(config, "chunk_size_feed_forward", 0) 
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num

        # Improved cross-attention logic
        if self.config.add_cross_attention:
            if self.config.cross_attention_freq == 1 or layer_num % self.config.cross_attention_freq == 0:
                self.crossattention = BertAttention(config, is_cross_attention=True)
                self.has_cross_attention = True
            else:
                self.has_cross_attention = False
        else:
            self.has_cross_attention = False

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        # Separate feed-forward layers for query processing
        if self.has_cross_attention:
            self.intermediate_query = BertIntermediate(config)
            self.output_query = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        query_length: int = 0,
    ) -> Tuple[torch.Tensor, ...]:
        # Self-attention
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        # Cross-attention
        if self.has_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states must be given for cross-attention layers")

            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            cross_attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]

            # Feed-forward for queries after cross-attention
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                cross_attention_output,
            )
        else:
            # Standard feed-forward
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )

        outputs = (layer_output,) + outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Standard feed-forward chunk."""
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Feed-forward chunk for query tokens after cross-attention."""
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    """
    Multi-layer BERT encoder for the Q-Former.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, i) for i in range(config.num_hidden_layers)])

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        device: torch.device,
        is_decoder: bool = False,
    ) -> torch.Tensor:
        """
        Prepares attention mask for broadcasting.
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask: {attention_mask.shape}")

        # Convert to float and apply large negative values to masked positions
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.to(device=device)
        
        return extended_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        query_length: int = 0,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None

        # Prepare attention masks
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, hidden_states.shape[:-1], hidden_states.device
            )

        extended_encoder_attention_mask = None
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            extended_encoder_attention_mask = self.get_extended_attention_mask(
                encoder_attention_mask, encoder_hidden_states.shape[:-1], encoder_hidden_states.device
            )

        # Process each layer
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    extended_encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    extended_encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ] if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class Qformer(nn.Module):
    """
    The Querying Transformer (Q-Former) module with improved device handling.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=config.initializer_range)

        self.bert_encoder = BertEncoder(config)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard practices."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(
        self,
        image_features: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with improved device and shape handling.
        """
        batch_size = image_features.shape[0]
        device = image_features.device

        # Ensure query tokens are on the correct device
        query_tokens = self.query_tokens.expand(batch_size, -1, -1).to(device)
        query_length = query_tokens.shape[1]

        # Create query attention mask
        query_attention_mask = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=device
        )

        # Handle image features shape
        if image_features.dim() == 2:
            # Single global feature per image: (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
            encoder_hidden_states = image_features.unsqueeze(1)
        else:
            # Multiple features per image: (batch_size, num_features, feature_dim)
            encoder_hidden_states = image_features

        # Create image attention mask if not provided
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                encoder_hidden_states.shape[:-1], dtype=torch.long, device=device
            )

        # Process through encoder
        encoder_outputs = self.bert_encoder(
            hidden_states=query_tokens,
            attention_mask=query_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
            query_length=query_length
        )

        # Return query embeddings
        return encoder_outputs.last_hidden_state