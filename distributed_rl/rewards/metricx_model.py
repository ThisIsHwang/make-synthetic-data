"""MT5-based regression model for MetricX scoring.

=== What is MetricX? ===

MetricX is a learned translation quality metric by Google.  It takes a
(source, candidate translation) pair (or source + candidate + reference)
and outputs a single scalar score predicting human judgment of quality.

The score scale is [0, 25]:
  - 0 = perfect translation
  - 25 = worst possible translation
  (Lower is better, opposite of what RL wants)

=== Model Architecture ===

MetricX is based on MT5 (Multilingual T5), an encoder-decoder transformer:

  1. ENCODER: Processes the input text:
     "source: Hello world candidate: 안녕하세요"
     → produces hidden states for each token

  2. DECODER: Takes a single zero token as input (dummy decoder input)
     and cross-attends to the encoder output.
     → produces one hidden state vector

  3. LM HEAD: Projects the decoder output to vocabulary logits.
     But instead of generating text, we extract the logit at a SPECIFIC
     token index (250089 = <extra_id_10>) and use it as the regression score.

  4. CLAMP: The raw logit is clamped to [0, 25] for numerical stability.

This is an unconventional use of a language model — the "prediction" is a
single logit value from the vocabulary projection, not a generated token.
The model was fine-tuned with MSE loss to make this logit match human scores.

=== Token ID 250089 ===

MT5's vocabulary has special tokens <extra_id_0> through <extra_id_99>.
<extra_id_10> has token ID 250089.  The MetricX authors chose this token
as the "output slot" for the regression score.  During fine-tuning, the
model learned to put the quality score in the logit at this position.

Ref: Direct copy from qwen3.5-35b-a3b/qwen35_moe_rl/metricx_model.py
Ref: MetricX paper — https://arxiv.org/abs/2401.06760
Ref: Upstream MetricX code — https://github.com/google-research/metricx
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Optional, Tuple, Union
import warnings

import torch
from torch import nn
import transformers
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

# Import MT5 internals from HuggingFace transformers.
# We use the internal MT5Stack and MT5PreTrainedModel classes because
# MetricX requires a custom forward() method that doesn't exist in
# the standard MT5ForConditionalGeneration.
MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack
_HEAD_MASK_WARNING_MSG = transformers.models.mt5.modeling_mt5.__HEAD_MASK_WARNING_MSG  # pylint: disable=protected-access


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    """Output of MT5ForRegression.forward().

    loss: MSE loss between predictions and labels (None if no labels provided).
    predictions: Tensor of shape (batch_size,) with quality scores in [0, 25].
    """
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor | None = None


class MT5ForRegression(MT5PreTrainedModel):
    """MT5 model adapted for regression (MetricX quality estimation).

    This is NOT a text generation model.  It's a regression model that:
      1. Encodes the input with the MT5 encoder
      2. Runs one step of the decoder with a dummy input (zero token)
      3. Extracts a single logit (at token ID 250089) as the quality score
      4. Clamps the score to [0, 25]

    The model weights are loaded from a fine-tuned MetricX checkpoint
    (e.g. "google/metricx-24-hybrid-xxl-v2p6").
    """

    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model  # Hidden size (e.g. 1024 for XL)

        # Shared embedding layer between encoder and decoder.
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # --- Encoder ---
        # Processes the input text ("source: X candidate: Y").
        # Configured as a non-decoder (no causal mask).
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        # --- Decoder ---
        # Cross-attends to encoder output with a single dummy token input.
        # This is the "aggregation" step that produces one vector summarizing
        # the encoder's understanding of the translation quality.
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        # LM head projects decoder hidden state → vocabulary logits.
        # We only use ONE logit (at index 250089) as the regression output.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights (HuggingFace standard).
        self.post_init()

        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MT5ForRegressionOutput]:
        """
        Forward pass for regression scoring.

        Args:
            input_ids: Tokenized input text (source + candidate translation).
            attention_mask: Mask for padding tokens.
            labels: Optional ground-truth quality scores for training (MSE loss).

        Returns:
            MT5ForRegressionOutput with ``predictions`` tensor of shape (batch_size,).
        """
        # Regression doesn't need KV caching (no autoregressive generation).
        use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # --- Step 1: Encode the input ---
        # Input: "source: Hello world candidate: 안녕하세요"
        # Output: hidden states for each token
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # --- Step 2: Run decoder with dummy input ---
        # The decoder input is a SINGLE zero token (decoder_input_ids = [[0]]).
        # This is a hack: the decoder cross-attends to encoder output and
        # produces one summary vector, which we then project to get the score.
        batch_size = input_ids.size(0) if input_ids is not None else hidden_states.size(0)
        decoder_device = hidden_states.device
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=decoder_device)
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=decoder_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output shape: (batch_size, 1, d_model)
        sequence_output = decoder_outputs[0]

        # --- Step 3: Project to vocabulary logits ---
        # If word embeddings are tied, scale by sqrt(d_model) inverse.
        # This is a T5 convention for tied embeddings.
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        # lm_logits shape: (batch_size, 1, vocab_size)
        lm_logits = self.lm_head(sequence_output)

        # --- Step 4: Extract regression score ---
        # Take the logit at vocabulary index 250089 (<extra_id_10>).
        # This specific index was chosen during MetricX fine-tuning.
        # lm_logits[:, 0, 250089] selects: all batches, first (only) position,
        # the <extra_id_10> logit.
        predictions = lm_logits[:, 0, 250089]

        # Clamp to [0, 25] — the valid MetricX score range.
        # Prevents numerical instability from producing out-of-range values.
        predictions = torch.clamp(predictions, 0, 25)

        # --- Optional: compute MSE loss for training ---
        # This is only used if fine-tuning MetricX itself (not our use case).
        # We only use this model for inference (scoring translations).
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            labels = labels.to(predictions.device)
            loss = loss_fct(predictions.view(-1), labels.view(-1))

        return MT5ForRegressionOutput(loss=loss, predictions=predictions)
