import math
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn

from cosmos_predict2.models.text2image_dit import Attention, GPT2FeedForward, PatchEmbed, RMSNorm, VideoRopePosition3DEmb


class TokenidxEmbed(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, use_adaln_lora: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self.get_embedding(num_embeddings, embedding_dim, padding_idx=None),
        )
        self.linear_1 = nn.Linear(embedding_dim, embedding_dim, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        else:
            self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.init_weights()

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.embedding_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.linear_2.weight)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sample = self.embedding[idx]
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


class VideoEmbed(nn.Module):
    def __init__(
        self,
        max_img_h: int = 240,
        max_img_w: int = 240,
        max_frames: int = 128,
        in_channels: int = 16,
        patch_spatial: int = 4,
        patch_temporal: int = 2,
        model_channels: int = 2048,
        num_heads: int = 16,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = False,
    ):
        super().__init__()
        self.patch_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )
        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=model_channels // num_heads,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_h_extrapolation_ratio,
            enable_fps_modulation=rope_enable_fps_modulation,
        )

    def init_weights(self) -> None:
        self.patch_embedder.init_weights()
        self.pos_embedder.reset_parameters()

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_B_C_T_H_W: Input tensor of shape (B, C, T, H, W).
            fps: Frames per second tensor of shape (B,).
        Returns:
            Output tensor of shape (B, C, T, H', W').
        """
        x_B_T_H_W_D = self.patch_embedder(x_B_C_T_H_W)
        rope_emb_L_1_1_D = self.pos_embedder(x_B_T_H_W_D, fps=fps)
        return x_B_T_H_W_D, rope_emb_L_1_1_D


class Block(nn.Module):
    def __init__(
        self,
        x_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "torch",
    ):
        super().__init__()
        self.x_dim = x_dim

        self.ln_attn_video = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)
        self.ln_attn_token = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)  # adaLN

        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads, qkv_format="bshd", backend=backend)

        self.ln_mlp_video = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)
        self.ln_mlp_token = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)  # adaLN

        self.mlp_video = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))
        self.mlp_token = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def reset_parameters(self) -> None:
        self.ln_attn_video.reset_parameters()
        self.ln_attn_token.reset_parameters()
        self.ln_mlp_video.reset_parameters()
        self.ln_mlp_token.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.mlp_video.init_weights()
        self.mlp_token.init_weights()

    def forward(
        self,
        token_B_T1_D: torch.Tensor,
        video_B_T2_D: torch.Tensor,
        emb_B_T1_D: torch.Tensor,
        rope_emb_T_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T1_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T1 = token_B_T1_D.shape[1]
        T2 = video_B_T2_D.shape[1]

        adaln_params_attn = self.adaln_modulation_self_attn(emb_B_T1_D)
        adaln_params_mlp = self.adaln_modulation_mlp(emb_B_T1_D)
        if self.use_adaln_lora and adaln_lora_B_T1_3D is not None:
            adaln_params_attn += adaln_lora_B_T1_3D
            adaln_params_mlp += adaln_lora_B_T1_3D
        shift_attn_B_T1_D, scale_attn_B_T1_D, gate_attn_B_T1_D = adaln_params_attn.chunk(3, dim=-1)
        shift_mlp_B_T1_D, scale_mlp_B_T1_D, gate_mlp_B_T1_D = adaln_params_mlp.chunk(3, dim=-1)

        token_B_T1_D = self.ln_attn_token(token_B_T1_D) * (1 + scale_attn_B_T1_D) + shift_attn_B_T1_D
        video_B_T2_D = self.ln_attn_video(video_B_T2_D)

        result_B_T_D = self.self_attn(torch.cat([token_B_T1_D, video_B_T2_D], dim=1), rope_emb=rope_emb_T_1_1_D)

        _token_B_T1_D, _video_B_T2_D = torch.split(result_B_T_D, [T1, T2], dim=1)
        token_B_T1_D = token_B_T1_D + _token_B_T1_D * gate_attn_B_T1_D
        video_B_T2_D = video_B_T2_D + _video_B_T2_D

        token_B_T1_D = self.ln_mlp_token(token_B_T1_D) * (1 + scale_mlp_B_T1_D) + shift_mlp_B_T1_D
        video_B_T2_D = self.ln_mlp_video(video_B_T2_D)

        token_B_T1_D = token_B_T1_D + self.mlp_token(token_B_T1_D) * gate_mlp_B_T1_D
        video_B_T2_D = video_B_T2_D + self.mlp_video(video_B_T2_D)

        return token_B_T1_D, video_B_T2_D


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        latent_action_dim: int,
        hidden_size: int = 768,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, latent_action_dim, bias=False)

        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        token_B_T_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        adaln_params = self.adaln_modulation(emb_B_T_D)
        if self.use_adaln_lora and adaln_lora_B_T_3D is not None:
            adaln_params += adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
        shift_B_T_D, scale_B_T_D = adaln_params.chunk(2, dim=-1)

        token_B_T_D = self.layer_norm(token_B_T_D) * (1 + scale_B_T_D) + shift_B_T_D

        token_B_T_D = self.linear(token_B_T_D)
        return token_B_T_D


class DualStreamTransformer(nn.Module):
    def __init__(
        self,
        latent_action_dim: int = 128,
        num_action_tokens: int = 8,
        max_img_h: int = 240,
        max_img_w: int = 240,
        max_frames: int = 128,
        in_channels: int = 16,
        patch_spatial: int = 4,
        patch_temporal: int = 2,
        model_channels: int = 768,
        num_heads: int = 12,
        num_blocks: int = 12,
        rope_h_extrapolation_ratio: float = 3.0,
        rope_w_extrapolation_ratio: float = 3.0,
        rope_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = False,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        backend: str = "torch",
    ):
        super().__init__()
        self.num_action_tokens = num_action_tokens
        self.model_channels = model_channels

        self.video_embedder = VideoEmbed(
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            max_frames=max_frames,
            in_channels=in_channels,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            model_channels=model_channels,
            num_heads=num_heads,
            rope_h_extrapolation_ratio=rope_h_extrapolation_ratio,
            rope_w_extrapolation_ratio=rope_w_extrapolation_ratio,
            rope_t_extrapolation_ratio=rope_t_extrapolation_ratio,
            rope_enable_fps_modulation=rope_enable_fps_modulation,
        )

        self.token_embedder = nn.Embedding(num_action_tokens, model_channels)
        self.token_idx_embedder = TokenidxEmbed(num_action_tokens, model_channels, use_adaln_lora)
        self.register_buffer("token_idx", torch.arange(num_action_tokens, dtype=torch.int64))
        self.register_buffer("token_dummy_rope_emb", torch.ones((num_action_tokens, 1, 1, model_channels // num_heads)))

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=backend,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            latent_action_dim=latent_action_dim,
            hidden_size=model_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )

        self.token_idx_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        self.init_weights()

    def init_weights(self) -> None:
        self.token_embedder.weight.data.uniform_(-1.0 / self.model_channels, 1.0 / self.model_channels)
        self.video_embedder.init_weights()
        self.token_idx_embedder.init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.token_idx_embedding_norm.reset_parameters()

    def forward(self, video_B_C_S_H_W: torch.Tensor, fps: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = video_B_C_S_H_W.shape[0]
        video_B_S_H_W_D, rope_emb_T2_1_1_D = self.video_embedder(video_B_C_S_H_W, fps=fps)
        rope_emb_T1_1_1_D = self.token_dummy_rope_emb

        token_T1_D = self.token_embedder(self.token_idx)
        token_B_T1_D = einops.repeat(token_T1_D, "T D -> B T D", B=B)
        video_B_T2_D = einops.rearrange(video_B_S_H_W_D, "B S H W D -> B (S H W) D")
        rope_emb_T_1_1_D = torch.cat([rope_emb_T1_1_1_D, rope_emb_T2_1_1_D], dim=0)

        emb_T1_D, adaln_lora_T1_3D = self.token_idx_embedder(self.token_idx)
        emb_B_T1_D = einops.repeat(emb_T1_D, "T D -> B T D", B=B)
        emb_B_T1_D = self.token_idx_embedding_norm(emb_B_T1_D)
        adaln_lora_B_T1_3D = einops.repeat(adaln_lora_T1_3D, "T D -> B T D", B=B)

        for block in self.blocks:
            token_B_T1_D, video_B_T2_D = block(
                token_B_T1_D,
                video_B_T2_D,
                emb_B_T1_D,
                rope_emb_T_1_1_D,
                adaln_lora_B_T1_3D,
            )

        token_B_T1_D = self.final_layer(token_B_T1_D, emb_B_T1_D, adaln_lora_B_T1_3D)

        return token_B_T1_D


if __name__ == "__main__":
    device = "cuda:0"
    video_B_C_S_H_W = torch.randn((2, 16, 6, 60, 60), device=device)
    m = DualStreamTransformer().to(device)

    token_B_T_D = m(video_B_C_S_H_W)
