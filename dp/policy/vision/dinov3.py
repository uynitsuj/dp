import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import AutoModel, AutoImageProcessor

# --------------------- utilities ---------------------

def _is_rgb_shape(shape: List[int]) -> bool:
    return isinstance(shape, (list, tuple)) and len(shape) == 3 and shape[0] in (1, 3)

def _rand_or_center_crop(x: torch.Tensor, crop_h: int, crop_w: int, center: bool) -> torch.Tensor:
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    if crop_h is None or crop_w is None or (crop_h == H and crop_w == W):
        return x
    if crop_h > H or crop_w > W:
        # fall back to resize if crop is larger than input
        return F.interpolate(x, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
    if center or not x.requires_grad:
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
    else:
        # deterministic in eval(), random in train()
        top = torch.randint(0, H - crop_h + 1, (1,), device=x.device).item()
        left = torch.randint(0, W - crop_w + 1, (1,), device=x.device).item()
    return x[:, :, top:top + crop_h, left:left + crop_w]

# --------------------- spatial softmax head ---------------------

class SpatialSoftmaxHead(nn.Module):
    def __init__(self, in_channels: int, num_kp: int = 32, temperature: float = 1.0):
        super().__init__()
        self.num_kp = num_kp
        self.temperature = temperature
        self.to_kp = nn.Conv2d(in_channels, num_kp, kernel_size=1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        logits = self.to_kp(feat) / max(self.temperature, 1e-6)   # (B, K, H, W)
        prob = logits.view(B, self.num_kp, H * W).softmax(dim=-1) # (B, K, HW)
        ys = torch.linspace(-1, 1, steps=H, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(-1, 1, steps=W, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).view(2, H * W)  # (2, HW)
        exp = torch.einsum("bkh,dh->bkd", prob, coords)               # (B, K, 2)
        return exp.reshape(B, self.num_kp * 2)                        # (B, 2K)

# --------------------- attention pooling head ---------------------


class AttnPoolHead(nn.Module):
    """
    Cross-attention pooling:
      - Input:  feat (B, D, gh, gw)
      - Queries: M learned vectors (M << gh*gw)
      - Attention: queries attend over all tokens (gh*gw)
      - Output:  pooled vector (B, out_dim)

    Complexity per batch ~ O(M * N * D), where N = gh*gw.

    Args:
      in_dim:       D (channel dim from backbone)
      out_dim:      output dim after pooling (e.g., 512)
      num_queries:  M (e.g., 4 or 8)
      num_heads:    attention heads (e.g., 8)
      ln_in:        apply LayerNorm on tokens/queries before attn
      ln_out:       apply LayerNorm on pooled output
      mlp:          optional MLP head after pooling
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        num_queries: int = 4,
        num_heads: int = 4,
        ln_in: bool = True,
        ln_out: bool = False,
        mlp: bool = False,
        mlp_hidden: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_queries = num_queries

        # Learned queries (M, D)
        self.query = nn.Parameter(torch.randn(num_queries, in_dim) * 0.02)

        # Cross-attention: Q (B, M, D), K/V (B, N, D)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.ln_tokens = nn.LayerNorm(in_dim) if ln_in else nn.Identity()
        self.ln_queries = nn.LayerNorm(in_dim) if ln_in else nn.Identity()

        # Project concatenated pooled queries (B, M*D) to out_dim
        self.proj = nn.Linear(num_queries * in_dim, out_dim)
        self.ln_out = nn.LayerNorm(out_dim) if ln_out else nn.Identity()

        if mlp:
            hid = mlp_hidden or max(out_dim, 256)
            self.mlp = nn.Sequential(
                nn.Linear(out_dim, hid),
                nn.GELU(),
                nn.Linear(hid, out_dim),
            )
        else:
            self.mlp = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, D, gh, gw) -> tokens: (B, N, D)
        B, D, gh, gw = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # (B, N, D)
        tokens = self.ln_tokens(tokens)

        # queries: (B, M, D)
        q = self.query.unsqueeze(0).expand(B, -1, -1).to(tokens.dtype)
        q = self.ln_queries(q)

        # Cross-attention: pooled_q = Attn(Q, K=tokens, V=tokens) -> (B, M, D)
        pooled_q, _ = self.attn(q, tokens, tokens, need_weights=False)

        # Concatenate M query outputs, project to out_dim
        pooled = pooled_q.reshape(B, self.num_queries * D)
        out = self.proj(pooled)
        out = self.ln_out(out)
        if self.mlp is not None:
            out = out + self.mlp(out)  # tiny residual MLP
        return out  # (B, out_dim)


# --------------------- DINOv3 conv-like adapter ---------------------

class Dinov3BackboneConvAdapter(nn.Module):
    """
    Loads a DINOv3 (or DINOv2 fallback) ViT backbone via HF Transformers and
    reshapes tokens to a (B, D, gh, gw) feature map. Drops cls token if present.
    """
    def __init__(self, model_name: str, target_size, freeze: bool, normalize_images: bool = True):
        super().__init__()
        self.target_size = target_size
        self.normalize_images = normalize_images

        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.backbone = None
        # Try DINOv3 first, then DINOv2, then generic AutoModel (trust_remote_code)
        model_loaded = False
        try:
            from transformers import DINOv3ViTModel  # type: ignore
            self.backbone = DINOv3ViTModel.from_pretrained(model_name)
            model_loaded = True
        except Exception:
            pass
        if not model_loaded:
            try:
                from transformers import Dinov2Model  # type: ignore
                self.backbone = Dinov2Model.from_pretrained(model_name)
                model_loaded = True
            except Exception:
                pass
        if not model_loaded:
            try:
                self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                model_loaded = True
            except Exception as e:
                raise RuntimeError(f"Could not obtain DINOv3/DINOv2 vision backbone: {e}")

        # Pull patch size and hidden size
        cfg = getattr(self.backbone, "config", None)
        vcfg = getattr(cfg, "vision_config", cfg) if cfg is not None else None

        patch_size = getattr(vcfg, "patch_size", 14)
        if isinstance(patch_size, (tuple, list)):
            patch_size = int(patch_size[0])
        self.patch_size = int(patch_size)

        # some configs use 'hidden_size' (ViT); others may use 'embed_dim'
        self.out_channels = int(getattr(vcfg, "hidden_size", getattr(vcfg, "embed_dim", 768)))

        # mean/std from the processor if present; fall back to ImageNet if missing
        mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        std  = getattr(self.processor, "image_std",  [0.229, 0.224, 0.225])
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor(std ).view(1, 3, 1, 1), persistent=False)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.float()

        # keep token grid consistent with ViT
        if (x.shape[-2], x.shape[-1]) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        if self.normalize_images:
            if x.max() > 1.0:
                x = x / 255.0
            x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

        outputs = self.backbone(pixel_values=x)
        tokens = getattr(outputs, "last_hidden_state", outputs[0])  # (B, N[, +1], D)

        H, W = self.target_size
        gh, gw = H // self.patch_size, W // self.patch_size
        B, N, D = tokens.shape

        # Drop any non-patch tokens that precede the patch grid (e.g., [CLS], register tokens)
        extra = N - (gh * gw)
        if extra > 0:
            tokens = tokens[:, extra:, :]
            N -= extra

        assert N == gh * gw, f"Tokens {N} != grid {gh}*{gw} (H={H}, W={W}, patch={self.patch_size})"
        return tokens.transpose(1, 2).contiguous().view(B, D, gh, gw)

# --------------------- image branch (crop -> DINO -> softmax) ---------------------

class Dinov3ImageBranch(nn.Module):
    def __init__(
        self,
        model_name: str,
        crop_size,
        target_size,
        freeze_backbone: bool,
        eval_fixed_crop: bool,
        normalize_images: bool,
        pool_mode: str = "attn",  # new arg
        pooled_dim: int = 512,
        num_queries: int = 8,
        attn_heads: int = 8,
    ):
        super().__init__()
        self.crop_h, self.crop_w = crop_size if crop_size is not None else (None, None)
        self.eval_fixed_crop = eval_fixed_crop
        self.backbone = Dinov3BackboneConvAdapter(
            model_name, target_size, freeze=freeze_backbone, normalize_images=normalize_images
        )
        D = self.backbone.out_channels

        assert pool_mode == "attn", "Set pool_mode='attn' for attention pooling"
        self.pool = AttnPoolHead(
            in_dim=D,
            out_dim=pooled_dim,
            num_queries=num_queries,
            num_heads=attn_heads,
            ln_in=True,
            ln_out=False,
            mlp=False,
        )

        # Optional: remember if backbone is frozen to skip grads
        self._frozen = all(not p.requires_grad for p in self.backbone.parameters())

    def forward(self, x):
        center = (not self.training) or self.eval_fixed_crop
        x = _rand_or_center_crop(x, self.crop_h, self.crop_w, center=center)

        requires_grad = any(p.requires_grad for p in self.backbone.parameters())
        if not requires_grad:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)

        return self.pool(feat)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
        # center = (not self.training) or self.eval_fixed_crop
        # x = _rand_or_center_crop(x, self.crop_h, self.crop_w, center=center)

        # if self._frozen:
        #     with torch.no_grad():
        #         feat = self.backbone(x)  # (B, D, gh, gw)
        # else:
        #     feat = self.backbone(x)
        # return self.pool(feat)            # (B, pooled_dim)

# --------------------- main dict encoder ---------------------

class Dinov3ObsEncoder(nn.Module):
    """
    Dict-in encoder:
      RGB keys -> Crop -> DINOv3 -> SpatialSoftmax(2K)
      Low-dim keys -> passthrough
    Matches the same interface your main() expects (constructor args, forward(), output_shape()).
    """
    def __init__(
        self,
        obs_key_shapes: Dict[str, List[int]],
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",  # try dinov3; falls back to dinov2 or trust_remote_code
        crop_size: Tuple[int, int] = (200, 200),
        target_size: Tuple[int, int] = (224, 224),
        num_kp: int = 32,
        freeze_backbone: bool = False,
        eval_fixed_crop: bool = False,
        normalize_images: bool = False,
    ):
        super().__init__()
        self.rgb_keys = [k for k, shp in obs_key_shapes.items() if _is_rgb_shape(shp)]
        self.low_dim_keys = [k for k, shp in obs_key_shapes.items() if not _is_rgb_shape(shp)]
        self.low_dim_dims = {k: int(obs_key_shapes[k][0]) for k in self.low_dim_keys}

        self.image_branches = nn.ModuleDict()
        for k in self.rgb_keys:
            self.image_branches[k] = Dinov3ImageBranch(
                model_name=model_name,
                crop_size=crop_size,
                target_size=target_size,
                num_kp=num_kp,
                freeze_backbone=freeze_backbone,
                eval_fixed_crop=eval_fixed_crop,
                normalize_images=normalize_images,
            )

        self.num_kp = num_kp
        self._out_dim = len(self.rgb_keys) * (2 * num_kp) + sum(self.low_dim_dims.values())

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        outs = []
        for k in self.rgb_keys:
            x = obs_dict[k]
            outs.append(self.image_branches[k](x))
        for k in self.low_dim_keys:
            x = obs_dict[k]
            if x.dim() > 2:
                x = x.view(x.shape[0], -1)
            outs.append(x)
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    def output_shape(self) -> List[int]:
        return [self._out_dim]

# --------------------- self-test ---------------------

if __name__ == "__main__":
    obs_key_shapes = {
        "left_camera-images-rgb":  [3, 224, 224],
        "right_camera-images-rgb": [3, 224, 224],
        "top_camera-images-rgb":   [3, 224, 224],
        "state":                   [14],
    }

    dinov3_obs_encoder = Dinov3ObsEncoder(
        obs_key_shapes=obs_key_shapes,
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",   # if this isn't available locally, try e.g. "facebook/dinov2-base"
        crop_size=(200, 200),
        target_size=(224, 224),
        num_kp=32,
        freeze_backbone=False,
        eval_fixed_crop=False,
        normalize_images=False,  # keep false if you already normalized upstream
    )

    B_flat = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov3_obs_encoder.to(device).train()

    dummy_obs = {
        "left_camera-images-rgb":  torch.rand(B_flat, 3, 224, 224, device=device),
        "right_camera-images-rgb": torch.rand(B_flat, 3, 224, 224, device=device),
        "top_camera-images-rgb":   torch.rand(B_flat, 3, 224, 224, device=device),
        "state":                   torch.randn(B_flat, 14,          device=device),
    }

    def _mem(device=None, prefix=""):
        if not torch.cuda.is_available():
            print(prefix + " CPU only"); return
        device = device or torch.device("cuda")
        a = torch.cuda.memory_allocated(device)
        r = torch.cuda.memory_reserved(device)
        m = torch.cuda.max_memory_allocated(device)
        to_gb = lambda x: f"{x/1024**3:.2f} GiB"
        print(f"{prefix} allocated={to_gb(a)} | reserved={to_gb(r)} | peak_allocated={to_gb(m)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    _mem(device, "[init] ")

    torch.cuda.synchronize(device); _mem(device, "[after build] ")

    out = dinov3_obs_encoder(dummy_obs)
    torch.cuda.synchronize(device); _mem(device, "[after fwd] ")

    print("output shape:", tuple(out.shape))

    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize(device); _mem(device, "[after bwd] ")

    for n, p in dinov3_obs_encoder.named_parameters():
        if p.requires_grad and p.grad is not None:
            print("grad ok on:", n, "|| norm:", float(p.grad.norm()))
            break