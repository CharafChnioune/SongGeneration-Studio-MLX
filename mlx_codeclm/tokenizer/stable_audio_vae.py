"""MLX port of the Stable Audio Tools VAE used by Flow1dVAE."""

from __future__ import annotations

import json
import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from mlx_codeclm.utils.module_list import ModuleList


def _pad1d(x: mx.array, pad: int) -> mx.array:
    if pad <= 0:
        return x
    return mx.pad(x, ((0, 0), (0, 0), (pad, pad)))


def _softplus(x: mx.array) -> mx.array:
    return mx.log1p(mx.exp(-mx.abs(x))) + mx.maximum(x, 0)


class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = mx.zeros((in_features,)) * alpha
            self.beta = mx.zeros((in_features,)) * alpha
        else:
            self.alpha = mx.ones((in_features,)) * alpha
            self.beta = mx.ones((in_features,)) * alpha
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        alpha = self.alpha[None, :, None]
        beta = self.beta[None, :, None]
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * mx.sin(x * alpha) ** 2


class WNConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.weight_g = mx.ones((out_channels, 1, 1))
        self.weight_v = mx.random.normal((out_channels, in_channels // groups, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight_v
        norm = mx.sqrt(mx.sum(weight * weight, axis=(1, 2), keepdims=True) + 1e-8)
        weight = weight * (self.weight_g / norm)
        weight = mx.transpose(weight, (0, 2, 1))
        x = mx.transpose(x, (0, 2, 1))
        y = mx.conv1d(x, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        y = mx.transpose(y, (0, 2, 1))
        if self.bias is not None:
            y = y + self.bias[None, :, None]
        return y


class WNConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.weight_g = mx.ones((in_channels, 1, 1))
        self.weight_v = mx.random.normal((in_channels, out_channels // groups, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight_v
        norm = mx.sqrt(mx.sum(weight * weight, axis=(1, 2), keepdims=True) + 1e-8)
        weight = weight * (self.weight_g / norm)
        weight = mx.transpose(weight, (1, 2, 0))
        x = mx.transpose(x, (0, 2, 1))
        y = mx.conv_transpose1d(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
            groups=self.groups,
        )
        y = mx.transpose(y, (0, 2, 1))
        if self.bias is not None:
            y = y + self.bias[None, :, None]
        return y


def get_activation(activation: str, channels: int) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int, use_snake: bool):
        super().__init__()
        activation = "snake" if use_snake else "elu"
        self.layers = ModuleList(
            [
                get_activation(activation, channels),
                WNConv1d(channels, channels, kernel_size=7, dilation=dilation, padding=(dilation * 6) // 2),
                get_activation(activation, channels),
                WNConv1d(channels, channels, kernel_size=1),
            ]
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool):
        super().__init__()
        activation = "snake" if use_snake else "elu"
        self.layers = ModuleList(
            [
                ResidualUnit(in_channels, dilation=1, use_snake=use_snake),
                ResidualUnit(in_channels, dilation=3, use_snake=use_snake),
                ResidualUnit(in_channels, dilation=9, use_snake=use_snake),
                get_activation(activation, in_channels),
                WNConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                ),
            ]
        )

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_snake: bool):
        super().__init__()
        activation = "snake" if use_snake else "elu"
        self.layers = ModuleList(
            [
                get_activation(activation, in_channels),
                WNConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                ),
                ResidualUnit(out_channels, dilation=1, use_snake=use_snake),
                ResidualUnit(out_channels, dilation=3, use_snake=use_snake),
                ResidualUnit(out_channels, dilation=9, use_snake=use_snake),
            ]
        )

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        latent_dim: int,
        c_mults: tp.Sequence[int],
        strides: tp.Sequence[int],
        use_snake: bool,
    ):
        super().__init__()
        c_mults = [1] + list(c_mults)
        layers = [WNConv1d(in_channels, c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1):
            layers.append(
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                )
            )
        activation = "snake" if use_snake else "elu"
        layers.append(get_activation(activation, c_mults[-1] * channels))
        layers.append(WNConv1d(c_mults[-1] * channels, latent_dim, kernel_size=3, padding=1))
        self.layers = ModuleList(layers)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        latent_dim: int,
        c_mults: tp.Sequence[int],
        strides: tp.Sequence[int],
        use_snake: bool,
        final_tanh: bool,
    ):
        super().__init__()
        c_mults = [1] + list(c_mults)
        layers = [WNConv1d(latent_dim, c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                )
            )
        activation = "snake" if use_snake else "elu"
        layers.append(get_activation(activation, c_mults[0] * channels))
        layers.append(WNConv1d(c_mults[0] * channels, out_channels, kernel_size=7, padding=3, bias=False))
        if final_tanh:
            layers.append(nn.Tanh())
        self.layers = ModuleList(layers)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def vae_sample(mean: mx.array, scale: mx.array) -> tuple[mx.array, mx.array]:
    stdev = _softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = mx.log(var)
    latents = mx.random.normal(mean.shape) * stdev + mean
    kl = mx.mean(mx.sum(mean * mean + var - logvar - 1.0, axis=1))
    return latents, kl


class VAEBottleneck(nn.Module):
    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        mean, scale = mx.split(x, 2, axis=1)
        return vae_sample(mean, scale)

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.__call__(x)

    def decode(self, x: mx.array) -> mx.array:
        return x


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        downsampling_ratio: int,
        sample_rate: int,
        io_channels: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = VAEBottleneck()
        self.latent_dim = latent_dim
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.in_channels = io_channels
        self.out_channels = io_channels

    def encode(self, audio: mx.array) -> mx.array:
        latents = self.encoder(audio)
        latents, _ = self.bottleneck.encode(latents)
        return latents

    def decode(self, latents: mx.array) -> mx.array:
        return self.decoder(self.bottleneck.decode(latents))

    def encode_audio(self, audio: mx.array, chunked: bool = False, overlap: int = 32, chunk_size: int = 128) -> mx.array:
        if not chunked:
            return self.encode(audio)
        samples_per_latent = self.downsampling_ratio
        total_size = audio.shape[2]
        batch_size = audio.shape[0]
        chunk_size_samples = chunk_size * samples_per_latent
        overlap_samples = overlap * samples_per_latent
        hop_size = chunk_size_samples - overlap_samples
        chunks = []
        i = 0
        for i in range(0, total_size - chunk_size_samples + 1, hop_size):
            chunks.append(audio[:, :, i : i + chunk_size_samples])
        if i + chunk_size_samples != total_size:
            chunks.append(audio[:, :, -chunk_size_samples:])
        chunks = mx.stack(chunks)
        num_chunks = chunks.shape[0]
        y_size = total_size // samples_per_latent
        y_final = mx.zeros((batch_size, self.latent_dim, y_size))
        for idx in range(num_chunks):
            y_chunk = self.encode(chunks[idx])
            if idx == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = idx * hop_size // samples_per_latent
                t_end = t_start + chunk_size_samples // samples_per_latent
            ol = overlap_samples // samples_per_latent // 2
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if idx > 0:
                t_start += ol
                chunk_start += ol
            if idx < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final

    def decode_audio(self, latents: mx.array, chunked: bool = False, overlap: int = 32, chunk_size: int = 128) -> mx.array:
        if not chunked:
            return self.decode(latents)
        hop_size = chunk_size - overlap
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        chunks = []
        i = 0
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunks.append(latents[:, :, i : i + chunk_size])
        if i + chunk_size != total_size:
            chunks.append(latents[:, :, -chunk_size:])
        chunks = mx.stack(chunks)
        num_chunks = chunks.shape[0]
        samples_per_latent = self.downsampling_ratio
        y_size = total_size * samples_per_latent
        y_final = mx.zeros((batch_size, self.out_channels, y_size))
        for idx in range(num_chunks):
            y_chunk = self.decode(chunks[idx])
            if idx == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = idx * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            ol = (overlap // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if idx > 0:
                t_start += ol
                chunk_start += ol
            if idx < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final


def create_autoencoder_from_config(model_config: dict) -> AudioAutoencoder:
    encoder_cfg = model_config["model"]["encoder"]["config"]
    decoder_cfg = model_config["model"]["decoder"]["config"]
    encoder = OobleckEncoder(
        in_channels=encoder_cfg["in_channels"],
        channels=encoder_cfg["channels"],
        latent_dim=encoder_cfg["latent_dim"],
        c_mults=encoder_cfg["c_mults"],
        strides=encoder_cfg["strides"],
        use_snake=encoder_cfg.get("use_snake", False),
    )
    decoder = OobleckDecoder(
        out_channels=decoder_cfg["out_channels"],
        channels=decoder_cfg["channels"],
        latent_dim=decoder_cfg["latent_dim"],
        c_mults=decoder_cfg["c_mults"],
        strides=decoder_cfg["strides"],
        use_snake=decoder_cfg.get("use_snake", False),
        final_tanh=decoder_cfg.get("final_tanh", False),
    )
    return AudioAutoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=model_config["model"]["latent_dim"],
        downsampling_ratio=model_config["model"]["downsampling_ratio"],
        sample_rate=model_config["sample_rate"],
        io_channels=model_config["model"]["io_channels"],
    )


def load_autoencoder(config_path: str, weights_path: str) -> AudioAutoencoder:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model = create_autoencoder_from_config(cfg)
    from mlx_codeclm.utils.weights import load_weights_npz

    load_weights_npz(model, weights_path)
    return model
