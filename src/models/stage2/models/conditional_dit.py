"""
条件DiT模型 - 用于BEV去噪
创建时间: 2025-10-18

基于LightningDiT，添加条件编码支持：
- 输入：噪声潜在表示作为条件
- 输出：干净潜在表示
- 方法：通过cross-attention融合条件信息
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp
from .model_utils import (
    VisionRotaryEmbeddingFast,
    SwiGLUFFN,
    RMSNorm,
    NormAttention,
    LabelEmbedder,
    get_2d_sincos_pos_embed,
    GaussianFourierEmbedding,
    modulate
)


class ConditionalDiTBlock(nn.Module):
    """
    条件DiT Block with cross-attention

    包含：
    - Self-attention (处理主输入)
    - Cross-attention (融合条件信息)
    - MLP
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()

        # Normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            self.norm3 = RMSNorm(hidden_size)

        # Self-attention
        self.self_attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # Cross-attention
        self.cross_attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # AdaLN modulation for self-attention and MLP
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)  # self-attn + cross-attn + mlp
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 9 * hidden_size, bias=True)  # shift + scale + gate for each
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, cond, feat_rope=None):
        """
        Args:
            x: 主输入 (B, N, D)
            c: 时间和类别嵌入 (B, D)
            cond: 条件编码 (B, M, D) - 来自噪声图的潜在表示
            feat_rope: Rotary position embedding
        """
        if self.wo_shift:
            # 6个参数：self-attn(scale, gate), cross-attn(scale, gate), mlp(scale, gate)
            params = self.adaLN_modulation(c).chunk(6, dim=1)
            scale_sa, gate_sa, scale_ca, gate_ca, scale_mlp, gate_mlp = params
            shift_sa = shift_ca = shift_mlp = None
        else:
            # 9个参数
            params = self.adaLN_modulation(c).chunk(9, dim=1)
            shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = params

        # Self-attention
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa),
            rope=feat_rope
        )

        # Cross-attention with condition
        # Query from x, Key and Value from cond
        x_norm = modulate(self.norm2(x), shift_ca, scale_ca)
        cross_out = self.cross_attn.forward_cross_attn(x_norm, cond)
        x = x + gate_ca.unsqueeze(1) * cross_out

        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )

        return x


class ConditionalNormAttention(nn.Module):
    """扩展的NormAttention，支持cross-attention"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        # 复用NormAttention的实现
        from .model_utils import NormAttention as BaseAttn
        self.attn = BaseAttn(*args, **kwargs)

        # 为cross-attention添加额外的kv投影
        dim = args[0] if len(args) > 0 else kwargs.get('dim', 768)
        self.cross_kv = nn.Linear(dim, dim * 2, bias=kwargs.get('qkv_bias', True))

    def forward(self, x, rope=None):
        """标准self-attention"""
        return self.attn(x, rope=rope)

    def forward_cross_attn(self, q, kv):
        """
        Cross-attention
        Args:
            q: query from main input (B, N, D)
            kv: key-value from condition (B, M, D)
        """
        B, N, D = q.shape
        _, M, _ = kv.shape

        # Query from q
        q_proj = self.attn.qkv.weight[:D, :] @ q.transpose(-2, -1) + self.attn.qkv.bias[:D].unsqueeze(-1)
        q_proj = q_proj.transpose(-2, -1).reshape(B, N, self.attn.num_heads, D // self.attn.num_heads).transpose(1, 2)

        # Key, Value from kv
        kv_proj = self.cross_kv(kv)
        k, v = kv_proj.chunk(2, dim=-1)
        k = k.reshape(B, M, self.attn.num_heads, D // self.attn.num_heads).transpose(1, 2)
        v = v.reshape(B, M, self.attn.num_heads, D // self.attn.num_heads).transpose(1, 2)

        # Attention
        attn = (q_proj @ k.transpose(-2, -1)) * (D // self.attn.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.attn.proj(out)

        return out


# 简化版：使用concatenation而不是cross-attention
class SimplifiedConditionalDiTBlock(nn.Module):
    """
    简化的条件DiT Block

    通过concatenation融合条件：
    - 将条件编码与输入拼接后处理
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()

        # Normalization
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Attention
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )

        # Condition projection
        self.cond_proj = nn.Linear(hidden_size, hidden_size)

        # AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, cond, feat_rope=None):
        """
        Args:
            x: 主输入 (B, N, D)
            c: 时间嵌入 (B, D)
            cond: 条件编码 (B, M, D)
        """
        # 将条件信息加到时间嵌入上
        cond_pooled = cond.mean(dim=1)  # (B, D)
        c_enhanced = c + self.cond_proj(cond_pooled)

        # 标准DiT block
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c_enhanced).chunk(4, dim=1)
            shift_msa = shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_enhanced).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class ConditionalDiT(nn.Module):
    """
    条件DiT模型用于BEV去噪

    接受噪声BEV的潜在表示作为条件，生成干净BEV的潜在表示
    """

    def __init__(
        self,
        input_size=16,
        patch_size=1,
        in_channels=768,
        hidden_size=384,  # 使用更小的hidden size for快速验证
        depth=12,  # 较浅的网络
        num_heads=6,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_cross_attn=False,  # 如果False，使用简化的concatenation方法
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_cross_attn = use_cross_attn

        # Embedders
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.cond_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # RoPE
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        # Transformer blocks
        BlockClass = SimplifiedConditionalDiTBlock if not use_cross_attn else ConditionalDiTBlock
        self.blocks = nn.ModuleList([
            BlockClass(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
            ) for _ in range(depth)
        ])

        # Final layer
        from .lightningDiT import LightningFinalLayer
        self.final_layer = LightningFinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.cond_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize embedders
        for embedder in [self.x_embedder, self.cond_embedder]:
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.proj.bias, 0)

        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x):
        """Convert patches back to image"""
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, cond):
        """
        Args:
            x: 噪声输入 (B, C, H, W) - 噪声化的干净潜在表示
            t: 时间步 (B,)
            cond: 条件 (B, C, H, W) - 来自噪声图的潜在表示

        Returns:
            预测的噪声或速度 (B, C, H, W)
        """
        # Embed inputs
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D)
        cond = self.cond_embedder(cond) + self.cond_pos_embed  # (B, M, D)

        # Time embedding
        t_emb = self.t_embedder(t)  # (B, D)

        # Process through blocks
        for block in self.blocks:
            x = block(x, t_emb, cond, feat_rope=self.feat_rope)

        # Final layer
        x = self.final_layer(x, t_emb)  # (B, N, patch_size^2 * C)
        x = self.unpatchify(x)  # (B, C, H, W)

        return x

    def forward_with_cfg(self, x, t, cond, cfg_scale):
        """
        Classifier-free guidance forward pass
        """
        # 暂时不实现CFG，直接返回条件预测
        return self.forward(x, t, cond)
