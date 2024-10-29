import torch
import torch.nn as nn
from  timm.utils import freeze
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial
from vit_dino import vit_base, trunc_normal_
import timm
import torch.nn.functional as F
import time


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)



class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class RankModel(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_dim, mlp_dim, n_patch, backbone='dinov1'):
        super().__init__()
        # self.backbone = ViTForImageClassification.from_pretrained("/apdcephfs_cq3/share_3208175/private/chengmingxu/model_weights/vit-mae-base", local_files_only=True)
        if backbone == 'dinov1':
            self.backbone = vit_base()
        elif backbone == 'clip':
            self.backbone = timm.create_model('vit_base_patch16_clip_224.openai', 
                              checkpoint_path='/apdcephfs_cq8/share_2992679/private/chengmingxu/model_weights/vit_base_patch16_clip_224.openai/pytorch_model.bin')
            self.backbone.reset_classifier(num_classes = 0)
        elif backbone == 'clip-l':
            try:
                self.backbone = timm.create_model('vit_large_patch14_clip_224.laion2b_ft_in12k_in1k', 
                          checkpoint_path='/apdcephfs_cq8/share_2992679/private/chengmingxu/model_weights/vit_large_patch14_clip_224.laion2b_ft_in12k/pytorch_model.bin')
            except:
                self.backbone = timm.create_model('vit_large_patch14_clip_224.laion2b_ft_in12k_in1k', 
                          checkpoint_path='/apdcephfs_cq10/share_1275017/chengmingxu/model_weights/vit_large_patch14_clip_224.laion2b_ft_in12k/pytorch_model.bin')                
            self.backbone.reset_classifier(num_classes = 0)
        elif backbone == 'dinov2':
            from dinov2.dinov2.hub.backbones import dinov2_vitb14
            self.backbone = dinov2_vitb14(pretrained=False)
        self.backbone.requires_grad_(False)

        self.head = []
        for _ in range(num_layers):
            self.head.append(EncoderBlock(num_heads, hidden_dim, mlp_dim, 0, 0))
        
        self.head = nn.Sequential(*self.head)

        self.cls_head = []
        for _ in range(2):
            self.cls_head.append(EncoderBlock(num_heads, hidden_dim, mlp_dim, 0, 0))
        
        self.cls_head = nn.Sequential(*self.head)
        self.predictor = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))

        self.n_patch = n_patch
        self.hidden_dim = hidden_dim

    def prepare_tokens(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], 1)

        return x

    def forward(self, query, gallery):
        '''
        query: batch_size, 3, h, w
        gallery: batch_size, num_gallery, 3, h, w
        '''
        # import pdb;pdb.set_trace()
        b, c, h, w = query.shape

        n_gallery = gallery.shape[1]
        gallery = gallery.view(-1, c, h, w)
        input_batch = torch.cat([query, gallery], 0)

        out = self.backbone.forward_features(input_batch)
        b_, n_patch, n_dim = out.shape
        img_cls_token, img_patch_token = out[:, 0].unsqueeze(1), out[:, 1:]

        w = int((n_patch-1)**0.5)
        w_ = int(w//2)

        # b_, 4, 768
        img_patch_token = F.avg_pool2d(img_patch_token.view(b_, w, w, n_dim).permute(0, 3, 1, 2), (w_, w_)).view(b_, n_dim, -1).permute(0, 2, 1)
 
        query_cls_token, gallery_cls_token = img_cls_token[:b], img_cls_token[b:]
        gallery_cls_token = gallery_cls_token.view(b, n_gallery, -1, n_dim).view(b, -1, n_dim)
        query_patch_token, gallery_patch_token = img_patch_token[:b], img_patch_token[b:]
        gallery_patch_token = gallery_patch_token.view(b, n_gallery, -1, n_dim).view(b, -1, n_dim)

        emb_seq = torch.cat([query_cls_token, gallery_cls_token, query_patch_token, gallery_patch_token], 1)

        emb_seq = self.head(emb_seq)
        emb_seq = self.cls_head(emb_seq[:, :1+n_gallery])[:, 1:]
        pred = self.predictor(emb_seq)
        return pred