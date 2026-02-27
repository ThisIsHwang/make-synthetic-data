from __future__ import annotations

import copy
import dataclasses
import inspect
from typing import Optional, Tuple, Union
import warnings

import torch
from torch import nn
import transformers
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack
_HEAD_MASK_WARNING_MSG = transformers.models.mt5.modeling_mt5.__HEAD_MASK_WARNING_MSG  # pylint: disable=protected-access


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor | None = None


class MT5ForRegression(MT5PreTrainedModel):
    """MetricX regression head on top of MT5.

    This mirrors the upstream MetricX implementation and predicts from the logit
    at token id 250089 (<extra_id_10>), clipped to [0, 25].
    """

    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = self._build_mt5_stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = self._build_mt5_stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.post_init()

        self.model_parallel = False
        self.device_map = None

    @staticmethod
    def _build_mt5_stack(config: MT5Config, shared: nn.Embedding) -> MT5Stack:
        """Build MT5Stack across transformers API variants.

        Some versions accept (config, embed_tokens), while others accept only
        (config) and require embeddings to be set separately.
        """
        try:
            params = inspect.signature(MT5Stack.__init__).parameters
            if "embed_tokens" in params:
                return MT5Stack(config, shared)
        except Exception:
            # Fallback to runtime probing below.
            pass

        try:
            stack = MT5Stack(config)
        except TypeError:
            # Older-style signature fallback.
            return MT5Stack(config, shared)

        if hasattr(stack, "set_input_embeddings"):
            stack.set_input_embeddings(shared)
        elif hasattr(stack, "embed_tokens"):
            stack.embed_tokens = shared
        return stack

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
        # MetricX regression uses a single-step decoder token, so caching is unnecessary
        # and can trigger shape mismatches across some transformers versions.
        use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(_HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

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

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # 250089 corresponds to <extra_id_10> in MT5 tokenizer.
        predictions = lm_logits[:, 0, 250089]
        predictions = torch.clamp(predictions, 0, 25)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            labels = labels.to(predictions.device)
            loss = loss_fct(predictions.view(-1), labels.view(-1))

        return MT5ForRegressionOutput(
            loss=loss,
            predictions=predictions,
        )
