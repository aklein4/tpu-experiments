import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from omegaconf import DictConfig

from torchprime.torch_xla_models.scan_layers import HomogeneousSequential

from models.llama import LlamaModel, LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm
from utils.torch_utils import (
    scale_gradient,
    expand_to_batch,
    unsqueeze_to_batch
)


class LoRaModulator(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        splits,
    ):
        super().__init__()

        self.base_linear = base_linear
        self.rank = rank
        self.splits = splits

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.num_splits = len(splits)
        self.total_rank = self.rank * self.num_splits

        self.lora_down = nn.Linear(
            self.in_features, self.total_rank, bias=False
        )
        self.lora_up = nn.Linear(
            self.total_rank, self.out_features, bias=False
        )

        # create the mask
        split_mask = torch.zeros(sum(splits), self.total_rank)

        row_start = 0
        col_start = 0
        for split_size in splits:

            split_mask[
                row_start:(row_start + split_size),
                col_start:(col_start + self.rank)
            ] = 1.0

            row_start += split_size
            col_start += self.rank

        self.register_buffer('split_mask', split_mask, persistent=False)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.base_linear(x)

        inner = (
            self.lora_down(x) * 
            unsqueeze_to_batch(self.split_mask, x)
        )
        outer = self.lora_up(inner)

        return (
            self.base_linear(x) * np.sqrt(0.5) +
            outer * np.sqrt(0.5)
        )


class ZRMEncoderLayer(nn.Module):

    def __init__(self, config, base_layer: LlamaDecoderLayer):
        super().__init__()

        self.hidden_size = base_layer.hidden_size

        self.self_attn = base_layer.self_attn

        self.mlp = base_layer.mlp
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm

        self.bi_attn_length = config.input_length + config.output_length
        self.bi_attn_norm = LlamaRMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps
        )
        self.bi_attn = LlamaAttention(
            config, layer_idx=self.self_attn.layer_idx, is_causal=False
        )
        self.bi_attn.qkv_proj = LoRaModulator(
            self.bi_attn.qkv_proj,
            config.lora_rank,
            [config.input_length, config.output_length]
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,    # necessary, but kept here for BC
        elementwise_attention_bias: torch.Tensor | None = None,
    ):

        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            elementwise_attention_bias=elementwise_attention_bias,
        )
        hidden_states = residual + hidden_states

        # Bidirectional Attention
        residual = hidden_states.clone()
        hidden_states = self.bi_attn_norm(hidden_states[:, :self.bi_attn_length])
        hidden_states = self.bi_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids[:, :self.bi_attn_length],
            position_embeddings=(p[:, :self.bi_attn_length] for p in position_embeddings),
            elementwise_attention_bias=elementwise_attention_bias[:, :self.bi_attn_length],
        )
        residual[:, :self.bi_attn_length] += hidden_states
        hidden_states = residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ZRMModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # transformer config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.lora_rank = config.lora_rank
        self.lr_scaler = np.sqrt(self.hidden_size)

        # length config
        self.input_length = config.input_length
        self.output_length = config.output_length
        self.z_length = config.z_length

        # transformers
        self.encoder = LlamaModel(config)
        self.generator = LlamaModel(config)
        self.decoder = LlamaModel(config)
        
        # add LoRa modulator to qkv and gate_up
        transformer_splits = [
            (
                self.encoder,
                [self.input_length, self.output_length, self.z_length]
            ),
            (
                self.generator,
                [self.input_length, self.z_length]
            ),
            (
                self.decoder,
                [self.input_length, self.z_length, self.output_length]
            ),
        ]
        for transformer, splits in transformer_splits:
            transformer: LlamaModel
            
            for layer in transformer.layers:
                layer: LlamaDecoderLayer

                layer.self_attn.qkv_proj = LoRaModulator(
                    layer.self_attn.qkv_proj,
                    self.lora_rank,
                    splits
                )
                layer.mlp.gate_up_proj = LoRaModulator(
                    layer.mlp.gate_up_proj,
                    self.lora_rank,
                    splits
                )
           
            transformer.embed_tokens = None

        self.encoder.layers = HomogeneousSequential(
            *[
                ZRMEncoderLayer(config, base_layer)
                for base_layer in self.encoder.layers
            ]
        )
        
        # LM components
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # input embeddings
        self.encoder_input_emb = nn.Parameter(
            torch.randn(1, self.hidden_size) / self.lr_scaler
        )
        self.encoder_sep_emb = nn.Parameter(
            torch.randn(self.hidden_size) / self.lr_scaler
        )
        self.encoder_output_emb = nn.Parameter(
            torch.randn(1, self.hidden_size) / self.lr_scaler
        )
        self.encoder_z_tokens = nn.Parameter(
            torch.randn(self.z_length, self.hidden_size) / self.lr_scaler
        )

        self.generator_input_emb = nn.Parameter(
            torch.randn(1, self.hidden_size) / self.lr_scaler
        )
        self.generator_z_tokens = nn.Parameter(
            torch.randn(self.z_length, self.hidden_size) / self.lr_scaler
        )

        self.decoder_start_z_token = nn.Parameter(
            torch.randn(self.hidden_size) / self.lr_scaler
        )
        self.decoder_z_tokens = nn.Parameter(
            torch.randn(self.z_length, self.hidden_size) / self.lr_scaler
        )
        self.decoder_input_emb = nn.Parameter(
            torch.randn(1, self.hidden_size) / self.lr_scaler
        )
        self.decoder_start_output_token = nn.Parameter(
            torch.randn(self.hidden_size) / self.lr_scaler
        )
        self.decoder_output_emb = nn.Parameter(
            torch.zeros(1, self.hidden_size) / self.lr_scaler
        )

        # z/noise io components
        self.encoder_noise_proj_in = nn.Linear(
            self.z_size, self.hidden_size, bias=False
        )
        self.encoder_base_mu_proj_out = nn.Linear(
            self.hidden_size, self.z_size, bias=False
        )
        self.encoder_extra_mu_proj_out = nn.Linear(
            self.hidden_size, self.z_size, bias=False
        )

        self.generator_z_proj_in = nn.Linear(
            self.z_size, self.hidden_size, bias=False
        )
        self.generator_mu_proj_out = nn.Linear(
            self.hidden_size, self.z_size, bias=False
        )

        self.decoder_z_proj_in = nn.Linear(
            self.z_size, self.hidden_size, bias=False
        )

        # scales to help with mu scaling
        self.mu_scale = np.sqrt(2 * np.log(self.vocab_size) * (self.output_length/self.z_length) / self.z_size)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)


    def _init_weights(self, module: nn.Module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1/module.in_features**0.5)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1/self.lr_scaler)


    def forward(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor,
        alpha: float = 0.0,
        noise_scale: float = 1.0,
        gen_grad_scale: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        assert input_ids.shape[-1] == self.input_length
        assert output_ids.shape[-1] == self.output_length

        # get reusable components
        input_tokens = self.embed_tokens(input_ids) * self.lr_scaler
        output_tokens = self.embed_tokens(output_ids) * self.lr_scaler

        input_mask = (input_ids != self.config.pad_token_id).float()
        output_mask = (output_ids != self.config.pad_token_id).float()

        input_bias = (input_ids == self.config.pad_token_id).float() * self.config.pad_bias
        output_bias = (output_ids == self.config.pad_token_id).float() * self.config.pad_bias

        # get the noise
        noise = torch.randn_like(
            input_tokens[:, :self.z_length, :self.z_size],
        ) * noise_scale

        # run the encoder
        encoder_mu_base, encoder_mu_extra = self.encode(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_mask=input_mask,
            output_mask=output_mask,
            input_bias=input_bias,
            output_bias=output_bias,
            noise=noise,
        )
        encoder_mu_extra = F.rms_norm(
            encoder_mu_extra, [self.z_size], eps=self.config.rms_norm_eps
        )
        encoder_mu_base = encoder_mu_base * self.mu_scale
        encoder_mu = (
            encoder_mu_base +
            alpha * encoder_mu_extra
        )

        # run the generator
        enc_mu_for_generator = scale_gradient(
            encoder_mu, gen_grad_scale
        )
        generator_mu = self.generate(
            input_tokens=input_tokens,
            input_mask=input_mask,
            input_bias=input_bias,
            z=(enc_mu_for_generator + noise)
        )
        generator_mu = generator_mu * self.mu_scale

        # run the decoder   
        output_logits = self.decode(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_mask=input_mask,
            output_mask=output_mask,
            input_bias=input_bias,
            output_bias=output_bias,
            z=(encoder_mu + noise)
        )

        return {
            "output_logits": output_logits,
            "encoder_mu": encoder_mu,
            "generator_mu": generator_mu,
            "encoder_mu_base": encoder_mu_base,
            "encoder_mu_extra": encoder_mu_extra,
        }
    

    def _shift_right(self, x, first=0.0):
        return torch.cat(
            [
                (x[:, :1] * 0) + first,
                x[:, :-1]
            ],
            dim=-2
        )
    

    def encode(
        self,
        input_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        output_mask: torch.Tensor,
        input_bias: torch.FloatTensor,
        output_bias: torch.FloatTensor,
        noise: torch.FloatTensor,
    ):
        
        # construct the encoder input
        input_states = (
            unsqueeze_to_batch(self.encoder_input_emb, input_tokens) * self.lr_scaler +
            input_tokens
        )

        output_states = (
            unsqueeze_to_batch(self.encoder_output_emb, output_tokens) * self.lr_scaler +
            torch.cat(
                [
                    output_tokens[:, :1] + unsqueeze_to_batch(self.encoder_sep_emb[None], output_tokens[:, :1]) * self.lr_scaler,
                    output_tokens[:, 1:],
                ],
                dim=-2
            )
        )

        print(self.encoder_z_tokens.shape, input_tokens.shape, noise.shape)
        z_states = (
            unsqueeze_to_batch(self.encoder_z_tokens, input_tokens) * self.lr_scaler +
            self.encoder_noise_proj_in(self._shift_right(noise))
        )

        encoder_states = torch.cat(
            [
                input_states,
                output_states,
                z_states,
            ],
            dim=-2
        )

        # create the position ids
        position_mask = torch.cat(
            [
                input_mask,
                output_mask,
                torch.ones_like(z_states[..., 0]),
            ],
            dim=-1
        )
        position_ids = position_mask.cumsum(dim=-1)

        # create the bias
        attention_bias = torch.cat(
            [
                input_bias,
                output_bias,
                torch.zeros_like(z_states[..., 0]),
            ],
            dim=-1
        )

        # run the encoder
        encoder_states = self.encoder(
            inputs_embeds=encoder_states,
            position_ids=position_ids,
            elementwise_attention_bias=attention_bias
        )
        
        # get the mu values
        mu_base = self.encoder_base_mu_proj_out(
            encoder_states[:, -self.z_length:]
        )
        mu_extra = self.encoder_extra_mu_proj_out(
            encoder_states[:, -self.z_length:]
        )

        return mu_base, mu_extra


    def generate(
        self,
        input_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        input_bias: torch.FloatTensor,
        z: torch.FloatTensor,
    ):
        z = F.rms_norm(
            z, [self.z_size],
            eps=self.config.rms_norm_eps
        )

        # construct the generator input
        input_states = (
            unsqueeze_to_batch(self.generator_input_emb, input_tokens) * self.lr_scaler +
            input_tokens
        )

        z_states = (
            unsqueeze_to_batch(self.generator_z_tokens, input_tokens) * self.lr_scaler +
            self.generator_z_proj_in(self._shift_right(z))
        )
 
        generator_states = torch.cat(
            [
                input_states,
                z_states,
            ],
            dim=-2
        )

        # create the position ids
        position_mask = torch.cat(
            [
                input_mask,
                torch.ones_like(z_states[..., 0]),
            ],
            dim=-1
        )
        position_ids = position_mask.cumsum(dim=-1)

        # create the bias
        attention_bias = torch.cat(
            [
                input_bias,
                torch.zeros_like(z_states[..., 0]),
            ],
            dim=-1
        )

        # run the generator
        generator_states = self.generator(
            inputs_embeds=generator_states,
            position_ids=position_ids,
            elementwise_attention_bias=attention_bias
        )
        
        # get the mu values
        mu = self.generator_mu_proj_out(
            generator_states[:, -self.z_length:]
        )

        return mu

    
    def decode(
        self,
        input_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        output_mask: torch.Tensor,
        input_bias: torch.FloatTensor,
        output_bias: torch.FloatTensor,
        z: torch.FloatTensor,
    ):
        z = F.rms_norm(
            z, [self.z_size],
            eps=self.config.rms_norm_eps
        )

        input_states = (
            unsqueeze_to_batch(self.decoder_input_emb, input_tokens) * self.lr_scaler +
            input_tokens
        )

        z_states = (
            unsqueeze_to_batch(self.decoder_z_tokens, z) * self.lr_scaler +
            self.decoder_z_proj_in(z)
        )

        output_states = (
            unsqueeze_to_batch(self.decoder_output_emb, output_tokens) * self.lr_scaler +
            self._shift_right(
                output_tokens,
                first=unsqueeze_to_batch(self.decoder_start_output_token[None], output_tokens[:, :1]) * self.lr_scaler
            )
        )

        decoder_states = torch.cat(
            [
                input_states,
                z_states,
                output_states,
            ],
            dim=-2
        )

        # create the position ids
        position_mask = torch.cat(
            [
                input_mask,
                torch.ones_like(z_states[..., 0]),
                torch.cat(
                    [
                        torch.ones_like(output_mask[..., :1]),
                        output_mask[..., :-1],
                    ],
                    dim=-1
                )
            ],
            dim=-1
        )
        position_ids = position_mask.cumsum(dim=-1)

        # create the bias
        attention_bias = torch.cat(
            [
                input_bias,
                torch.zeros_like(z_states[..., 0]),
                torch.cat(
                    [
                        torch.zeros_like(output_bias[..., :1]),
                        output_bias[..., :-1],
                    ],
                    dim=-1
                )
            ],
            dim=-1
        )

        # run the decoder
        decoder_states = self.decoder(
            inputs_embeds=decoder_states,
            position_ids=position_ids,
            elementwise_attention_bias=attention_bias,
        )
        
        # get the lm head logits
        output_logits = self.lm_head(decoder_states[:, -self.output_length:])

        return output_logits
    