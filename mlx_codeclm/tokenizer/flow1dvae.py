"""MLX port of Flow1dVAE tokenizer (decode path)."""

from __future__ import annotations

import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from mlx_codeclm.tokenizer.gpt2_reflow import GPT2Model
from mlx_codeclm.tokenizer.rvq import ResidualVectorQuantize
from mlx_codeclm.tokenizer.stable_audio_vae import AudioAutoencoder, load_autoencoder
from mlx_codeclm.utils.weights import load_weights_npz


def normalize_prompt_audio(prompt: tp.Optional[mx.array], sample_rate: int) -> tp.Optional[mx.array]:
    if prompt is None:
        return None
    prompt = mx.array(prompt)
    if prompt.ndim == 3:
        prompt = prompt[0]
    if prompt.ndim == 1:
        prompt = prompt[None, :]
    if prompt.shape[0] == 1:
        prompt = mx.concatenate([prompt, prompt], axis=0)
    elif prompt.shape[0] > 2:
        prompt = prompt[:2, :]
    if prompt.shape[-1] < int(30 * sample_rate):
        prompt = prompt[:, : int(10 * sample_rate)]
    else:
        prompt = prompt[:, int(20 * sample_rate) : int(30 * sample_rate)]
    return prompt


class Feature1DProcessor(nn.Module):
    def __init__(self, dim: int = 64, power_std: float = 1.0, num_samples: int = 100_000, cal_num_frames: int = 600):
        super().__init__()
        self.num_samples = num_samples
        self.dim = dim
        self.power_std = power_std
        self.cal_num_frames = cal_num_frames
        self.counts = mx.zeros((1,))
        self.sum_x = mx.zeros((dim,))
        self.sum_x2 = mx.zeros((dim,))
        self.sum_target_x2 = mx.zeros((dim,))

    @property
    def mean(self) -> mx.array:
        if self.counts.item() < 10:
            return mx.zeros_like(self.sum_x)
        return self.sum_x / self.counts

    @property
    def std(self) -> mx.array:
        if self.counts.item() < 10:
            return mx.ones_like(self.sum_x)
        var = self.sum_x2 / self.counts - self.mean**2
        return mx.sqrt(mx.maximum(var, 0))

    def project_sample(self, x: mx.array) -> mx.array:
        if self.counts.item() < self.num_samples:
            self.counts = self.counts + x.shape[0]
            sample = x[:, :, : self.cal_num_frames]
            self.sum_x = self.sum_x + mx.sum(mx.mean(sample, axis=2), axis=0)
            self.sum_x2 = self.sum_x2 + mx.sum(mx.mean(sample**2, axis=2), axis=0)
        rescale = (1.0 / mx.maximum(self.std, 1e-12)) ** self.power_std
        return (x - self.mean[None, :, None]) * rescale[None, :, None]

    def return_sample(self, x: mx.array) -> mx.array:
        rescale = self.std ** self.power_std
        return x * rescale[None, :, None] + self.mean[None, :, None]


class BaseCFM(nn.Module):
    def __init__(self, estimator: GPT2Model, mlp: nn.Module):
        super().__init__()
        self.estimator = estimator
        self.mlp = mlp
        self.sigma_min = 1e-4

    def solve_euler(
        self,
        x: mx.array,
        latent_mask_input: mx.array,
        incontext_x: mx.array,
        incontext_length: int,
        t_span: mx.array,
        mu: mx.array,
        attention_mask: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        dt = t_span[1:] - t_span[:-1]
        t = t_span[:-1]
        bsz = x.shape[0]

        if guidance_scale > 1.0:
            attention_mask = mx.concatenate([attention_mask, attention_mask], axis=0)

        x_next = x
        noise = x
        for i in range(dt.shape[0]):
            ti = t[i]
            if incontext_length > 0:
                prefix = (1 - (1 - self.sigma_min) * ti) * noise[:, :incontext_length] + ti * incontext_x[:, :incontext_length]
                x_next = mx.concatenate([prefix, x_next[:, incontext_length:]], axis=1)

            if guidance_scale > 1.0:
                def double(z):
                    return mx.concatenate([z, z], axis=0)

                model_input = mx.concatenate(
                    [
                        double(latent_mask_input),
                        double(incontext_x),
                        mx.concatenate([mx.zeros_like(mu), mu], axis=0),
                        double(x_next),
                    ],
                    axis=2,
                )
                timestep = mx.full((2 * bsz,), ti)
            else:
                model_input = mx.concatenate([latent_mask_input, incontext_x, mu, x_next], axis=2)
                timestep = mx.full((bsz,), ti)

            v = self.estimator(inputs_embeds=model_input, attention_mask=attention_mask, time_step=timestep).last_hidden_state
            v = v[..., -x.shape[2] :]

            if guidance_scale > 1.0:
                v_uncond, v_cond = mx.split(v, 2, axis=0)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            x_next = x_next + dt[i] * v

        return x_next


class PromptCondAudioDiffusion(nn.Module):
    def __init__(self, num_channels: int, gpt_config: dict):
        super().__init__()
        self.num_channels = num_channels
        self.normfeat = Feature1DProcessor(dim=64)
        self.rvq_bestrq_emb = ResidualVectorQuantize(
            input_dim=1024,
            n_codebooks=1,
            codebook_size=16_384,
            codebook_dim=32,
            quantizer_dropout=0.0,
            stale_tolerance=200,
        )
        self.zero_cond_embedding1 = mx.random.normal((32 * 32,))
        self.mask_emb = nn.Embedding(3, 48)
        estimator = GPT2Model(
            vocab_size=gpt_config["vocab_size"],
            max_position_embeddings=gpt_config["max_position_embeddings"],
            hidden_size=gpt_config["hidden_size"],
            num_hidden_layers=gpt_config["num_hidden_layers"],
            num_attention_heads=gpt_config["num_attention_heads"],
            n_inner=gpt_config["n_inner"],
            activation_function=gpt_config["activation_function"],
            layer_norm_epsilon=gpt_config.get("layer_norm_epsilon", 1e-5),
        )
        mlp = nn.Sequential(
            nn.Linear(gpt_config["mlp_in"], gpt_config["mlp_hidden"]),
            nn.SiLU(),
            nn.Linear(gpt_config["mlp_hidden"], gpt_config["mlp_hidden"]),
            nn.SiLU(),
            nn.Linear(gpt_config["mlp_hidden"], gpt_config["mlp_out"]),
        )
        self.cfm_wrapper = BaseCFM(estimator, mlp)

    def inference_codes(
        self,
        codes: tp.List[mx.array],
        true_latents: mx.array,
        latent_length: int,
        incontext_length: int = 127,
        guidance_scale: float = 2.0,
        num_steps: int = 20,
        scenario: str = "start_seg",
    ) -> mx.array:
        codes_bestrq_emb = codes[0]
        batch_size = codes_bestrq_emb.shape[0]
        quantized_bestrq_emb, _, _ = self.rvq_bestrq_emb.from_codes(codes_bestrq_emb)
        quantized_bestrq_emb = mx.transpose(quantized_bestrq_emb, (0, 2, 1))

        num_frames = quantized_bestrq_emb.shape[1]
        latents = mx.random.normal((batch_size, num_frames, 64))

        idx = mx.arange(num_frames)
        latent_masks = mx.where(idx < latent_length, 2, 0)
        latent_masks = mx.broadcast_to(latent_masks[None, :], (batch_size, num_frames))
        if scenario == "other_seg":
            incontext_length = min(int(incontext_length), int(latent_length))
            if incontext_length > 0:
                incontext_idx = mx.where(idx < incontext_length, 1, latent_masks[0])
                latent_masks = mx.broadcast_to(incontext_idx[None, :], (batch_size, num_frames))

        mask = (latent_masks > 0.5).astype(quantized_bestrq_emb.dtype)
        quantized_bestrq_emb = mask[:, :, None] * quantized_bestrq_emb + (1.0 - mask[:, :, None]) * self.zero_cond_embedding1[None, None, :]

        true_latents = mx.transpose(true_latents, (0, 2, 1))
        true_latents = self.normfeat.project_sample(true_latents)
        true_latents = mx.transpose(true_latents, (0, 2, 1))
        incontext_mask = ((latent_masks > 0.5) & (latent_masks < 1.5)).astype(true_latents.dtype)
        incontext_latents = true_latents * incontext_mask[:, :, None]
        incontext_length = int(mx.sum(incontext_mask[0]).item())

        attention_mask = (latent_masks > 0.5).astype(mx.float32)
        attention_mask = attention_mask[:, None, :]
        attention_mask = attention_mask * mx.transpose(attention_mask, (0, 2, 1))
        attention_mask = attention_mask[:, None, :, :]

        latent_mask_input = self.mask_emb(latent_masks.astype(mx.int32))
        additional_model_input = quantized_bestrq_emb

        t_span = mx.linspace(0, 1, num_steps + 1)
        latents = self.cfm_wrapper.solve_euler(
            latents,
            latent_mask_input,
            incontext_latents,
            incontext_length,
            t_span,
            additional_model_input,
            attention_mask,
            guidance_scale,
        )

        if incontext_length > 0:
            latents = mx.concatenate([incontext_latents[:, :incontext_length], latents[:, incontext_length:]], axis=1)
        latents = mx.transpose(latents, (0, 2, 1))
        latents = self.normfeat.return_sample(latents)
        return latents


class Flow1dVAE1rvq(nn.Module):
    def __init__(self, vae_config: str, vae_weights: str, gpt_config: dict, weights_path: str):
        super().__init__()
        self.sample_rate = 48000
        self.vae = load_autoencoder(vae_config, vae_weights)
        self.model = PromptCondAudioDiffusion(num_channels=32, gpt_config=gpt_config)
        load_weights_npz(
            self.model,
            weights_path,
            quiet=True,
            ignore_prefixes=(
                "bestrq.",
                "hubert.",
                "rsp48toclap",
                "rsq48tobestrq",
                "rsq48tohubert",
                "rsq48towav2vec",
            ),
        )

    def decode(
        self,
        codes: mx.array,
        prompt: tp.Optional[mx.array] = None,
        prompt_codes: tp.Optional[mx.array] = None,
        guidance_scale: float = 1.5,
        num_steps: int = 50,
        duration: tp.Optional[float] = None,
        chunked: bool = False,
        chunk_size: int = 128,
    ):
        codes = codes.astype(mx.int32)
        if duration is None:
            duration = 40.0
        min_frames = max(1, int(duration * 25))
        hop_frames = max(1, (min_frames // 4) * 3)
        ovlp_frames = min_frames - hop_frames
        orig_codes_len = codes.shape[-1]
        first_latent = mx.random.normal((codes.shape[0], min_frames, 64))
        first_latent_length = 0
        first_latent_codes_length = 0
        if prompt is not None:
            prompt = normalize_prompt_audio(prompt, self.sample_rate)
            true_latent = self.vae.encode_audio(prompt[None, ...])
            true_latent = mx.transpose(true_latent, (0, 2, 1))
            first_latent[:, : true_latent.shape[1], :] = true_latent
            first_latent_length = true_latent.shape[1]
            if prompt_codes is not None:
                if prompt_codes.ndim == 1:
                    prompt_codes = prompt_codes[None, None, :]
                elif prompt_codes.ndim == 2:
                    prompt_codes = prompt_codes[:, None, :]
                if prompt_codes.shape[0] != codes.shape[0]:
                    prompt_codes = mx.broadcast_to(prompt_codes, (codes.shape[0],) + prompt_codes.shape[1:])
                if prompt_codes.shape[1] != codes.shape[1]:
                    prompt_codes = mx.broadcast_to(prompt_codes, (codes.shape[0], codes.shape[1], prompt_codes.shape[2]))
                prompt_codes = prompt_codes.astype(mx.int32)
                first_latent_codes_length = prompt_codes.shape[-1]
                codes = mx.concatenate([prompt_codes, codes], axis=-1)
        codes_len = codes.shape[-1]
        target_len = int((codes_len - first_latent_codes_length) * self.sample_rate / 25)
        if codes_len < min_frames:
            while codes.shape[-1] < min_frames:
                codes = mx.concatenate([codes, codes], axis=-1)
            codes = codes[:, :, :min_frames]
        codes_len = codes.shape[-1]
        if (codes_len - ovlp_frames) % hop_frames > 0:
            len_codes = math.ceil((codes_len - ovlp_frames) / float(hop_frames)) * hop_frames + ovlp_frames
            while codes.shape[-1] < len_codes:
                codes = mx.concatenate([codes, codes], axis=-1)
            codes = codes[:, :, :len_codes]
        latent_length = min_frames
        latent_list = []
        for sinx in range(0, codes.shape[-1] - hop_frames, hop_frames):
            codes_input = [codes[:, :, sinx : sinx + min_frames]]
            if sinx == 0:
                incontext_length = first_latent_length
                latents = self.model.inference_codes(
                    codes_input,
                    first_latent,
                    latent_length,
                    incontext_length=incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    scenario="other_seg",
                )
            else:
                prev_latents = latent_list[-1]
                true_latent = mx.transpose(prev_latents[:, :, -ovlp_frames:], (0, 2, 1))
                incontext_length = true_latent.shape[1]
                pad_len = min_frames - incontext_length
                if pad_len > 0:
                    pad_noise = mx.random.normal((true_latent.shape[0], pad_len, true_latent.shape[2]))
                    true_latent = mx.concatenate([true_latent, pad_noise], axis=1)
                latents = self.model.inference_codes(
                    codes_input,
                    true_latent,
                    latent_length,
                    incontext_length=incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    scenario="other_seg",
                )
            latent_list.append(latents)

        if first_latent_length > 0 and latent_list:
            latent_list[0] = latent_list[0][:, :, first_latent_length:]

        samples_per_frame = (self.sample_rate // 1000) * 40
        min_samples = min_frames * samples_per_frame
        hop_samples = hop_frames * samples_per_frame
        ovlp_samples = min_samples - hop_samples
        output = None
        for latent in latent_list:
            cur_output = self.vae.decode_audio(latent, chunked=chunked, chunk_size=chunk_size)[0]
            if output is None:
                output = cur_output
            else:
                if ovlp_samples > 0:
                    ov = mx.linspace(0.0, 1.0, ovlp_samples)
                    ov_win = mx.concatenate([ov[None, :], (1.0 - ov)[None, :]], axis=-1)
                    blend = output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:] + cur_output[:, :ovlp_samples] * ov_win[
                        :, :ovlp_samples
                    ]
                    output = mx.concatenate([output[:, :-ovlp_samples], blend, cur_output[:, ovlp_samples:]], axis=-1)
                else:
                    output = mx.concatenate([output, cur_output], axis=-1)
        if output is None:
            raise ValueError("No decoded segments produced")
        output = output[:, :target_len]
        return output[None, ...]


class PromptCondAudioDiffusionSep(nn.Module):
    def __init__(self, num_channels: int, gpt_config: dict):
        super().__init__()
        self.num_channels = num_channels
        self.normfeat = Feature1DProcessor(dim=64)
        self.rvq_bestrq_emb = ResidualVectorQuantize(
            input_dim=1024,
            n_codebooks=1,
            codebook_size=16_384,
            codebook_dim=32,
            quantizer_dropout=0.0,
            stale_tolerance=200,
        )
        self.rvq_bestrq_bgm_emb = ResidualVectorQuantize(
            input_dim=1024,
            n_codebooks=1,
            codebook_size=16_384,
            codebook_dim=32,
            quantizer_dropout=0.0,
            stale_tolerance=200,
        )
        self.zero_cond_embedding1 = mx.random.normal((32 * 32,))
        self.mask_emb = nn.Embedding(3, 24)
        estimator = GPT2Model(
            vocab_size=gpt_config["vocab_size"],
            max_position_embeddings=gpt_config["max_position_embeddings"],
            hidden_size=gpt_config["hidden_size"],
            num_hidden_layers=gpt_config["num_hidden_layers"],
            num_attention_heads=gpt_config["num_attention_heads"],
            n_inner=gpt_config["n_inner"],
            activation_function=gpt_config["activation_function"],
            layer_norm_epsilon=gpt_config.get("layer_norm_epsilon", 1e-5),
        )
        mlp = nn.Sequential(
            nn.Linear(gpt_config["mlp_in"], gpt_config["mlp_hidden"]),
            nn.SiLU(),
            nn.Linear(gpt_config["mlp_hidden"], gpt_config["mlp_hidden"]),
            nn.SiLU(),
            nn.Linear(gpt_config["mlp_hidden"], gpt_config["mlp_out"]),
        )
        self.cfm_wrapper = BaseCFM(estimator, mlp)

    def inference_codes(
        self,
        codes: tp.List[mx.array],
        true_latents: mx.array,
        latent_length: int,
        incontext_length: int = 127,
        guidance_scale: float = 2.0,
        num_steps: int = 20,
        scenario: str = "start_seg",
    ) -> mx.array:
        codes_vocal, codes_bgm = codes
        batch_size = codes_vocal.shape[0]
        quant_vocal, _, _ = self.rvq_bestrq_emb.from_codes(codes_vocal)
        quant_bgm, _, _ = self.rvq_bestrq_bgm_emb.from_codes(codes_bgm)
        quant_vocal = mx.transpose(quant_vocal, (0, 2, 1))
        quant_bgm = mx.transpose(quant_bgm, (0, 2, 1))

        num_frames = quant_vocal.shape[1]
        latents = mx.random.normal((batch_size, num_frames, 64))

        idx = mx.arange(num_frames)
        latent_masks = mx.where(idx < latent_length, 2, 0)
        latent_masks = mx.broadcast_to(latent_masks[None, :], (batch_size, num_frames))
        if scenario == "other_seg":
            incontext_length = min(int(incontext_length), int(latent_length))
            if incontext_length > 0:
                incontext_idx = mx.where(idx < incontext_length, 1, latent_masks[0])
                latent_masks = mx.broadcast_to(incontext_idx[None, :], (batch_size, num_frames))

        mask = (latent_masks > 0.5).astype(quant_vocal.dtype)
        quant_vocal = mask[:, :, None] * quant_vocal + (1.0 - mask[:, :, None]) * self.zero_cond_embedding1[None, None, :]
        quant_bgm = mask[:, :, None] * quant_bgm + (1.0 - mask[:, :, None]) * self.zero_cond_embedding1[None, None, :]

        true_latents = mx.transpose(true_latents, (0, 2, 1))
        true_latents = self.normfeat.project_sample(true_latents)
        true_latents = mx.transpose(true_latents, (0, 2, 1))
        incontext_mask = ((latent_masks > 0.5) & (latent_masks < 1.5)).astype(true_latents.dtype)
        incontext_latents = true_latents * incontext_mask[:, :, None]
        incontext_length = int(mx.sum(incontext_mask[0]).item())

        attention_mask = (latent_masks > 0.5).astype(mx.float32)
        attention_mask = attention_mask[:, None, :]
        attention_mask = attention_mask * mx.transpose(attention_mask, (0, 2, 1))
        attention_mask = attention_mask[:, None, :, :]

        latent_mask_input = self.mask_emb(latent_masks.astype(mx.int32))
        additional_model_input = mx.concatenate([quant_vocal, quant_bgm], axis=2)

        t_span = mx.linspace(0, 1, num_steps + 1)
        latents = self.cfm_wrapper.solve_euler(
            latents,
            latent_mask_input,
            incontext_latents,
            incontext_length,
            t_span,
            additional_model_input,
            attention_mask,
            guidance_scale,
        )

        if incontext_length > 0:
            latents = mx.concatenate([incontext_latents[:, :incontext_length], latents[:, incontext_length:]], axis=1)
        latents = mx.transpose(latents, (0, 2, 1))
        latents = self.normfeat.return_sample(latents)
        return latents


class Flow1dVAESeparate(nn.Module):
    def __init__(self, vae_config: str, vae_weights: str, gpt_config: dict, weights_path: str):
        super().__init__()
        self.sample_rate = 48000
        self.vae = load_autoencoder(vae_config, vae_weights)
        self.model = PromptCondAudioDiffusionSep(num_channels=32, gpt_config=gpt_config)
        load_weights_npz(
            self.model,
            weights_path,
            quiet=True,
            ignore_prefixes=(
                "bestrq.",
                "hubert.",
                "rsp48toclap",
                "rsq48tobestrq",
                "rsq48tohubert",
                "rsq48towav2vec",
            ),
        )

    def decode(
        self,
        codes: tp.Union[tp.Sequence[mx.array], mx.array],
        prompt_vocal: tp.Optional[mx.array] = None,
        prompt_bgm: tp.Optional[mx.array] = None,
        prompt_codes: tp.Optional[tp.Sequence[mx.array]] = None,
        guidance_scale: float = 1.5,
        num_steps: int = 50,
        duration: tp.Optional[float] = None,
        chunked: bool = False,
        chunk_size: int = 128,
    ):
        if isinstance(codes, (list, tuple)):
            codes_vocal, codes_bgm = codes
        else:
            codes_vocal = codes[:, [0], :]
            codes_bgm = codes[:, [1], :]
        codes_vocal = codes_vocal.astype(mx.int32)
        codes_bgm = codes_bgm.astype(mx.int32)

        if duration is None:
            duration = 40.0
        min_frames = max(1, int(duration * 25))
        hop_frames = max(1, (min_frames // 4) * 3)
        ovlp_frames = min_frames - hop_frames

        prompt_vocal_norm = None
        prompt_bgm_norm = None
        first_latent_codes_length = 0
        if prompt_vocal is not None and prompt_bgm is not None:
            prompt_vocal_norm = normalize_prompt_audio(prompt_vocal, self.sample_rate)
            prompt_bgm_norm = normalize_prompt_audio(prompt_bgm, self.sample_rate)
            if prompt_codes is not None:
                prompt_codes_vocal, prompt_codes_bgm = prompt_codes
                if prompt_codes_vocal.ndim == 1:
                    prompt_codes_vocal = prompt_codes_vocal[None, None, :]
                elif prompt_codes_vocal.ndim == 2:
                    prompt_codes_vocal = prompt_codes_vocal[:, None, :]
                if prompt_codes_bgm.ndim == 1:
                    prompt_codes_bgm = prompt_codes_bgm[None, None, :]
                elif prompt_codes_bgm.ndim == 2:
                    prompt_codes_bgm = prompt_codes_bgm[:, None, :]
                if prompt_codes_vocal.shape[0] != codes_vocal.shape[0]:
                    prompt_codes_vocal = mx.broadcast_to(
                        prompt_codes_vocal, (codes_vocal.shape[0],) + prompt_codes_vocal.shape[1:]
                    )
                if prompt_codes_bgm.shape[0] != codes_bgm.shape[0]:
                    prompt_codes_bgm = mx.broadcast_to(
                        prompt_codes_bgm, (codes_bgm.shape[0],) + prompt_codes_bgm.shape[1:]
                    )
                prompt_codes_vocal = prompt_codes_vocal.astype(mx.int32)
                prompt_codes_bgm = prompt_codes_bgm.astype(mx.int32)
                first_latent_codes_length = prompt_codes_vocal.shape[-1]
                codes_vocal = mx.concatenate([prompt_codes_vocal, codes_vocal], axis=-1)
                codes_bgm = mx.concatenate([prompt_codes_bgm, codes_bgm], axis=-1)

        codes_len = codes_vocal.shape[-1]
        target_len = int((codes_len - first_latent_codes_length) * self.sample_rate / 25)

        if codes_len < min_frames:
            while codes_vocal.shape[-1] < min_frames:
                codes_vocal = mx.concatenate([codes_vocal, codes_vocal], axis=-1)
                codes_bgm = mx.concatenate([codes_bgm, codes_bgm], axis=-1)
            codes_vocal = codes_vocal[:, :, :min_frames]
            codes_bgm = codes_bgm[:, :, :min_frames]
        codes_len = codes_vocal.shape[-1]
        if (codes_len - ovlp_frames) % hop_frames > 0:
            len_codes = math.ceil((codes_len - ovlp_frames) / float(hop_frames)) * hop_frames + ovlp_frames
            while codes_vocal.shape[-1] < len_codes:
                codes_vocal = mx.concatenate([codes_vocal, codes_vocal], axis=-1)
                codes_bgm = mx.concatenate([codes_bgm, codes_bgm], axis=-1)
            codes_vocal = codes_vocal[:, :, :len_codes]
            codes_bgm = codes_bgm[:, :, :len_codes]

        latent_length = min_frames
        latent_list = []
        first_latent = mx.random.normal((codes_vocal.shape[0], min_frames, 64))
        first_latent_length = 0
        if prompt_vocal_norm is not None and prompt_bgm_norm is not None:
            true_latent = self.vae.encode_audio((prompt_vocal_norm + prompt_bgm_norm)[None, ...])
            true_latent = mx.transpose(true_latent, (0, 2, 1))
            first_latent[:, : true_latent.shape[1], :] = true_latent
            first_latent_length = true_latent.shape[1]
        for sinx in range(0, codes_vocal.shape[-1] - hop_frames, hop_frames):
            codes_input = [
                codes_vocal[:, :, sinx : sinx + min_frames],
                codes_bgm[:, :, sinx : sinx + min_frames],
            ]
            if sinx == 0:
                incontext_length = first_latent_length
                latents = self.model.inference_codes(
                    codes_input,
                    first_latent,
                    latent_length,
                    incontext_length=incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    scenario="other_seg",
                )
            else:
                prev_latents = latent_list[-1]
                true_latent = mx.transpose(prev_latents[:, :, -ovlp_frames:], (0, 2, 1))
                incontext_length = true_latent.shape[1]
                pad_len = min_frames - incontext_length
                if pad_len > 0:
                    pad_noise = mx.random.normal((true_latent.shape[0], pad_len, true_latent.shape[2]))
                    true_latent = mx.concatenate([true_latent, pad_noise], axis=1)
                latents = self.model.inference_codes(
                    codes_input,
                    true_latent,
                    latent_length,
                    incontext_length=incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    scenario="other_seg",
                )
            latent_list.append(latents)

        if first_latent_length > 0 and latent_list:
            latent_list[0] = latent_list[0][:, :, first_latent_length:]

        samples_per_frame = (self.sample_rate // 1000) * 40
        min_samples = min_frames * samples_per_frame
        hop_samples = hop_frames * samples_per_frame
        ovlp_samples = min_samples - hop_samples

        output = None
        for latent in latent_list:
            cur_output = self.vae.decode_audio(latent, chunked=chunked, chunk_size=chunk_size)[0]
            if output is None:
                output = cur_output
            else:
                if ovlp_samples > 0:
                    ov = mx.linspace(0.0, 1.0, ovlp_samples)
                    ov_win = mx.concatenate([ov[None, :], (1.0 - ov)[None, :]], axis=-1)
                    blend = output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:] + cur_output[:, :ovlp_samples] * ov_win[
                        :, :ovlp_samples
                    ]
                    output = mx.concatenate([output[:, :-ovlp_samples], blend, cur_output[:, ovlp_samples:]], axis=-1)
                else:
                    output = mx.concatenate([output, cur_output], axis=-1)

        if output is None:
            raise ValueError("No decoded segments produced")
        output = output[:, :target_len]
        return output[None, ...]
