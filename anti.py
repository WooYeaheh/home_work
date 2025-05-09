import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import torchvision
import CLIP.clip
from CLIP.clip.model import build_model
from collections import defaultdict
from torchvision import transforms
import matplotlib.pyplot as plt

from torch import nn, Tensor
from typing import Any, Callable, List, Optional
from torchvision.models.swin_transformer import ShiftedWindowAttention, SwinTransformerBlock

def min_max_normalize(t):
    t_min = t.min()
    t_max = t.max()
    return (t - t_min) / (t_max - t_min + 1e-8)

import inspect

def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    valid_keys = sig.parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valid_keys}

class swin_ca(ShiftedWindowAttention):
    def __init__(self,dim, window_size, # : List[int],
        shift_size,#: List[int],
        num_heads,#: int,
        attention_dropout,#: float = 0.0,
        dropout,**kwargs):
        super().__init__(dim,
        window_size, # : List[int],
        shift_size,#: List[int],
        num_heads,#: int,
        attention_dropout,#: float = 0.0,
        dropout,**kwargs)#: float = 0.0,)

        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
    def shifted_window_attention(self,
            input_q: Tensor,
            input_k: Tensor,
            input_v: Tensor,
            relative_position_bias: Tensor,
            window_size: List[int],
            num_heads: int,
            shift_size: List[int],
            attention_dropout: float = 0.0,
            dropout: float = 0.0,
            logit_scale: Optional[torch.Tensor] = None,
            training: bool = True,
    ) -> Tensor:

        B, H, W, C = input_q.shape
        # pad feature maps to multiples of window size
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        x_q = F.pad(input_q, (0, 0, 0, pad_r, 0, pad_b))
        x_k = F.pad(input_k, (0, 0, 0, pad_r, 0, pad_b))
        x_v = F.pad(input_v, (0, 0, 0, pad_r, 0, pad_b))

        _, pad_H, pad_W, _ = x_q.shape

        shift_size = shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window
        if window_size[0] >= pad_H:
            shift_size[0] = 0
        if window_size[1] >= pad_W:
            shift_size[1] = 0

        # cyclic shift
        if sum(shift_size) > 0:
            x_q = torch.roll(x_q, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            x_k = torch.roll(x_k, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            x_v = torch.roll(x_v, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
        x_q = x_q.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        x_q = x_q.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        x_k = x_k.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        x_k = x_k.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)
        x_v = x_v.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        x_v = x_v.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)

        # multi-head attention
        q = self.q(x_q).reshape(x_v.size(0), x_v.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
        k = self.k(x_k).reshape(x_v.size(0), x_v.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
        v = self.v(x_v).reshape(x_v.size(0), x_v.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)

        if logit_scale is not None:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
        else:
            q = q * (C // num_heads) ** -0.5
            attn = q.matmul(k.transpose(-2, -1))
        # add relative position bias
        attn = attn + relative_position_bias

        if sum(shift_size) > 0:
            # generate attention mask
            attn_mask = x_q.new_zeros((pad_H, pad_W))
            h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
            w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0]: h[1], w[0]: w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x_q.size(0) // num_windows, num_windows, num_heads, x_q.size(1), x_q.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, num_heads, x_q.size(1), x_q.size(1))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=attention_dropout, training=training)

        x = attn.matmul(v).transpose(1, 2).reshape(x_q.size(0), x_q.size(1), C)
        x = self.proj(x)
        x = F.dropout(x, p=dropout, training=training)

        # reverse windows
        x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()

        return x

    def forward(self, x_q, x_k, x_v):
        relative_position_bias = self.get_relative_position_bias()

        return self.shifted_window_attention(
            x_q, x_k, x_v,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            training=self.training,
        )

class Swin_CA_block(SwinTransformerBlock):
    def __init__(self,
        attn_layer,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer = nn.LayerNorm,
        **kwargs):
        super().__init__(
        dim,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochastic_depth_prob,
        norm_layer,
        attn_layer,
    )
    def forward(self, x_q, x_k, x_v):
        x = x_q + self.stochastic_depth(self.attn(self.norm1(x_q),self.norm1(x_k),self.norm1(x_v)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x

class Swin_anti(nn.Module):
    ''' 2 2 8 2 block들 사이에만 adapter '''
    def __init__(self):
        super().__init__()
        from torchvision.models.swin_transformer import swin_b, shifted_window_attention
        self.model = swin_b(pretrained = True)
        for p in self.model.parameters():
            p.requires_grad = False
        # for p in self.model.features[-1].parameters():
        #     p.requires_grad = True
        self.model.head = nn.Linear(self.model.head.in_features,2)
        depths = self.model.depths
        self.num_layers = len(self.model.features) # 8

        self.Patch_merge = nn.ModuleList()
        for idx in range(self.num_layers//2):
            self.Patch_merge.append(self.model.features[2*idx])

        self.Swin_blocks_group = nn.ModuleList()
        for idx in range(self.num_layers//2):
            swin_blocks = nn.Sequential()
            for i_layers in range(depths[idx]):
                swin_blocks.append(self.model.features[2*idx+1][i_layers]) # Swinblock들
            self.Swin_blocks_group.append(swin_blocks) # 1,56,56,128

        self.CA_blocks_1 = nn.ModuleList()
        for idx in range(self.num_layers//2):
            kwargs = self.Swin_blocks_group[idx][0]._init_args
            self.CA_blocks_1.append(swin_ca(**kwargs)) # Swin_CA_block
        self.CA_blocks_2 = copy.deepcopy(self.CA_blocks_1)

        super_dict = {k: v for k, v in self.model.named_modules() if k in ['permute', 'norm','avgpool','flatten','head']}
        for k, v in super_dict.items():
            if k != 'self':
                setattr(self, k, v)

    def forward(self, rgb, ir):
        output = defaultdict(torch.tensor)
        for idx in range(self.num_layers//2):
            rgb = self.Patch_merge[idx](rgb)
            ir = self.Patch_merge[idx](ir)

            x_r = self.Swin_blocks_group[idx](rgb)
            x_i = self.Swin_blocks_group[idx](ir)

            rgb = x_r + self.CA_blocks_1[idx](x_r, x_i, x_i)
            ir = x_i + self.CA_blocks_2[idx](x_i, x_r, x_r)
        feat1 = self.forward_last(rgb)
        feat2 = self.forward_last(ir)
        output['feat_1'], output['rgb_feats'] = feat1
        output['feat_2'], output['ir_feats'] = feat2
        return output
    def forward_last(self, x):
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        feats = self.flatten(x)
        x = self.head(feats)
        return x, feats

class Swin_anti_adapter1(nn.Module):
    '''
    if mlp_ratio = 4.0 (default)
        just Swin params : 87.7M
        params : 296M
        trainable params : 235M
    elif mlp_ratio = 0.5 (revised version)
        trainable params : 110M

    Adapter를 swinblock -> swin ca로 변경
        trainable params : 97M
    '''
    def __init__(self):
        super().__init__()
        from torchvision.models.swin_transformer import swin_b, shifted_window_attention
        self.model = swin_b(pretrained = True)
        for p in self.model.parameters():
            p.requires_grad = False
        # for p in self.model.features[-1].parameters():
        #     p.requires_grad = True
        self.model.head = nn.Linear(self.model.head.in_features,2)
        self.depths = self.model.depths
        self.num_layers = len(self.model.features) # 8

        self.Patch_merge = nn.ModuleList()
        for idx in range(self.num_layers//2):
            self.Patch_merge.append(self.model.features[2*idx])

        self.Swin_blocks_group = nn.ModuleList()
        for idx in range(self.num_layers//2):
            swin_blocks = nn.Sequential()
            for i_layers in range(self.depths[idx]):
                swin_blocks.append(self.model.features[2*idx+1][i_layers]) # Swinblock들
            self.Swin_blocks_group.append(swin_blocks) # 1,56,56,128

        self.Swin_blocks_Adapters1 = nn.ModuleList()
        for idx in range(self.num_layers // 2):
            swin_adapters = nn.Sequential()
            for i_layers in range(self.depths[idx]):
                kwargs = self.model.features[2 * idx + 1][i_layers]._init_args
                swin_adapters.append(swin_ca(**kwargs))
            self.Swin_blocks_Adapters1.append(swin_adapters)  # 1,56,56,128

        self.Swin_blocks_Adapters2 = copy.deepcopy(self.Swin_blocks_Adapters1)

        super_dict = {k: v for k, v in self.model.named_modules() if k in ['permute', 'norm','avgpool','flatten','head']}
        for k, v in super_dict.items():
            if k != 'self':
                setattr(self, k, v)

    def forward(self, rgb, ir):
        output = defaultdict(torch.tensor)
        for idx in range(self.num_layers//2):
            rgb = self.Patch_merge[idx](rgb)
            ir = self.Patch_merge[idx](ir)

            for i_layer in range(self.depths[idx]):

                x_r = self.Swin_blocks_group[idx][i_layer](rgb)
                x_i = self.Swin_blocks_group[idx][i_layer](ir)

                rgb = x_r + self.Swin_blocks_Adapters1[idx][i_layer](x_r, x_i, x_i)
                ir = x_i + self.Swin_blocks_Adapters2[idx][i_layer](x_i, x_r, x_r)
        feat1 = self.forward_last(rgb)
        feat2 = self.forward_last(ir)
        output['feat_1'], output['rgb_feats'] = feat1
        output['feat_2'], output['ir_feats'] = feat2
        return output

    def forward_last(self, x):
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        feats = self.flatten(x)
        x = self.head(feats)
        return x, feats


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CDC_Adapter(nn.Module):
    def __init__(self, adapterdim=8, theta=0.7):
        super(CDC_Adapter, self).__init__()

        self.adapter_conv = Conv2d_cd(in_channels=adapterdim, out_channels=adapterdim, kernel_size=3, stride=1,
                                      padding=1, theta=theta)

        # nn.init.xavier_uniform_(self.adapter_conv.weight)

        self.adapter_down = nn.Linear(768, adapterdim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(adapterdim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = adapterdim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        # pdb.set_trace()

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up

class my_ViT_3modality_CDC_Adapter(nn.Module):
    def __init__(self, ):
        super(my_ViT_3modality_CDC_Adapter, self).__init__()
        self.num_encoders = 12
        dim = 768

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = True
        self.conv_proj2 = copy.deepcopy(self.conv_proj1)
        self.conv_proj3 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)
        self.class_token3 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1  # 패치개수+1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding3 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim + dim, 2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim, 2),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(dim, 2),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(dim, 2),
        )
        self.adapter_1_1, self.adapter_1_2 = [], []
        self.adapter_2_1, self.adapter_2_2 = [], []
        self.adapter_3_1, self.adapter_3_2 = [], []

        for i in range(self.num_encoders):
            self.adapter_1_1.append(CDC_Adapter())
            self.adapter_1_2.append(CDC_Adapter())
            self.adapter_2_1.append(CDC_Adapter())
            self.adapter_2_2.append(CDC_Adapter())
            self.adapter_3_1.append(CDC_Adapter())
            self.adapter_3_2.append(CDC_Adapter())

        self.adapter_1_1 = nn.Sequential(*self.adapter_1_1)
        self.adapter_2_1 = nn.Sequential(*self.adapter_2_1)
        self.adapter_3_1 = nn.Sequential(*self.adapter_3_1)
        self.adapter_1_2 = nn.Sequential(*self.adapter_1_2)
        self.adapter_2_2 = nn.Sequential(*self.adapter_2_2)
        self.adapter_3_2 = nn.Sequential(*self.adapter_3_2)

        self.out = None

    def forward(self, x1, x2, x3):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x3 = self.conv_proj3(x3)  # b,d,gh,gw
        x3 = x3.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d
        x3 = torch.cat((self.class_token3.expand(b, -1, -1), x3), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2
        proj3 = x3 + self.pos_embedding3

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)
        proj3 = self.ViT_Encoder[0](proj3)

        for i in range(1, min(len(self.ViT_Encoder), len(self.ViT_Encoder)) - 1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x3 = self.ViT_Encoder[i].ln_1(proj3)

            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)
            x3_c, _ = self.ViT_Encoder[i].self_attention(x3, x3, x3, need_weights=False)

            x1_c = self.ViT_Encoder[i].dropout(x1_c)
            x2_c = self.ViT_Encoder[i].dropout(x2_c)
            x3_c = self.ViT_Encoder[i].dropout(x3_c)

            x1 = proj1 + x1_c + self.ViT_Encoder[i].ln_1(self.adapter_1_1[i - 1](x1))
            x2 = proj2 + x2_c + self.ViT_Encoder[i].ln_1(self.adapter_2_1[i - 1](x2))
            x3 = proj3 + x3_c + self.ViT_Encoder[i].ln_1(self.adapter_3_1[i - 1](x3))

            y1 = self.ViT_Encoder[i].ln_2(x1)
            y2 = self.ViT_Encoder[i].ln_2(x2)
            y3 = self.ViT_Encoder[i].ln_2(x3)

            y1_m = self.ViT_Encoder[i].mlp(y1)
            y2_m = self.ViT_Encoder[i].mlp(y2)
            y3_m = self.ViT_Encoder[i].mlp(y3)

            proj1 = x1 + y1_m + self.ViT_Encoder[i].ln_2(self.adapter_1_2[i - 1](y1))
            proj2 = x2 + y2_m + self.ViT_Encoder[i].ln_2(self.adapter_2_2[i - 1](y2))
            proj3 = x3 + y3_m + self.ViT_Encoder[i].ln_2(self.adapter_3_2[i - 1](y3))

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)
        proj3 = self.ViT_Encoder[-1](proj3)

        logits1 = proj1[:, 0]  # b,d
        out1 = self.fc1(logits1)  # b,num_classes
        logits2 = proj2[:, 0]  # b,d
        out2 = self.fc2(logits2)  # b,num_classes
        logits3 = proj3[:, 0]  # b,d
        out3 = self.fc3(logits3)  # b,num_classes
        logits_all = torch.cat([logits1, logits2, logits3], dim=1)
        out_all = self.fc_all(logits_all)
        self.out = {
            "out_all": out_all,
            "out_1": out1,
            "out_2": out2,
            "out_3": out3
        }
        return self.out

    def cal_loss(self, spoof_label, loss_func):
        loss_global = loss_func(self.out['out_all'], spoof_label.squeeze(-1))
        loss_1 = loss_func(self.out['out_1'], spoof_label.squeeze(-1))
        loss_2 = loss_func(self.out['out_2'], spoof_label.squeeze(-1))
        loss_3 = loss_func(self.out['out_3'], spoof_label.squeeze(-1))
        loss = loss_global + loss_1 + loss_2 + loss_3
        return loss

class My_MHA_UC(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_k, x_v):
        B, N, C = x_q.shape # b 1 d
        B2, N2, C2 = x_k.shape
        q = self.q_linear(x_q).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x_k).reshape(B, N2, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x_v).reshape(B, N2, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class U_Adapter(nn.Module):
    def __init__(self, adapter_dim=8, theta=0.5, use_cdc=True, dropout_rate=0.2, hidden_dim=768):
        super(U_Adapter, self).__init__()
        if use_cdc:
            self.adapter_conv = Conv2d_cd(in_channels=adapter_dim, out_channels=adapter_dim,
                                          kernel_size=3, stride=1, padding=1, theta=theta)
        else:
            self.adapter_conv = nn.Conv2d(in_channels=adapter_dim, out_channels=adapter_dim,
                                          kernel_size=3, stride=1, padding=1)
        self.cross_attention = My_MHA_UC(dim=adapter_dim)
        self.ln_before = nn.LayerNorm(adapter_dim)
        # nn.init.xavier_uniform_(self.adapter_conv.conv.weight)  # CDC xavier初始化
        # nn.init.zeros_(self.adapter_conv.conv.bias)

        self.adapter_down_1 = nn.Linear(hidden_dim, adapter_dim)  # equivalent to 1 * 1 Conv
        self.adapter_down_2 = nn.Linear(hidden_dim, adapter_dim)  # equivalent to 1 * 1 Conv

        self.adapter_up = nn.Linear(adapter_dim, hidden_dim)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down_1.weight)
        nn.init.zeros_(self.adapter_down_1.bias)
        nn.init.xavier_uniform_(self.adapter_down_2.weight)
        nn.init.zeros_(self.adapter_down_2.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dim = adapter_dim

    def forward(self, x, x_c):
        B, N, C = x.shape

        x_down_1 = self.adapter_down_1(x)  # equivalent to 1 * 1 Conv
        x_down_1 = self.act(x_down_1)

        x_down_2 = self.adapter_down_2(x_c)  # equivalent to 1 * 1 Conv
        x_down_2 = self.act(x_down_2)

        x_cross = self.cross_attention(x_down_2, x_down_1, x_down_1)  #2,1
        x_down = self.ln_before(x_cross + x_down_1)

        x_patch = x_down[:, 1:(1 + 14 * 14)]
        x_patch = x_patch.reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch = self.adapter_conv(x_patch)

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up

class MMDG(nn.Module):
    '''
    trainable params : 7.5M
    '''
    def __init__(self, dropout_ratio=0.3, exp_rate=2.25, num_sample=20, adapter_dim=8, r_ssp=0.3):
        super(MMDG, self).__init__()
        self.num_encoders = 5
        dim = 768

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter_1_2_2 = []
        self.adapter_2_1_2 = []

        for i in range(self.num_encoders):
            self.adapter_1_2_2.append(U_Adapter(adapter_dim=adapter_dim))
            self.adapter_2_1_2.append(U_Adapter(adapter_dim=adapter_dim))

        self.adapter_1_2_2 = nn.Sequential(*self.adapter_1_2_2)
        self.adapter_2_1_2 = nn.Sequential(*self.adapter_2_1_2)

        self.out = None

    def forward(self, x1, x2, domain=None):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d


        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        for i in range(1, len(self.ViT_Encoder) - 1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            y1 = self.ViT_Encoder[i].ln_2(x1_a)
            y2 = self.ViT_Encoder[i].ln_2(x2_a)
            y1_m = self.ViT_Encoder[i].mlp(y1)
            y2_m = self.ViT_Encoder[i].mlp(y2)
            proj1 = x1_a + y1_m + self.adapter_1_2_2[i - 1](y1, y2)
            proj2 = x2_a + y2_m + self.adapter_2_1_2[i - 1](y2, y1)

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class ConvAdapter(nn.Module):
    def __init__(self, adapter_dim=8, theta=0.5, use_cdc=True, dropout_rate=0.2, hidden_dim=768):
        super(ConvAdapter, self).__init__()
        self.adapter_conv = Conv2d_cd(in_channels=adapter_dim, out_channels=adapter_dim,
                                          kernel_size=3, stride=1, padding=1, theta=theta)

        self.adapter_down = nn.Linear(hidden_dim, adapter_dim)

        self.adapter_up = nn.Linear(adapter_dim, hidden_dim)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dim = adapter_dim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:(1 + 14 * 14)]
        x_patch = x_patch.reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch = self.adapter_conv(x_patch)

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up

class MMDG2(nn.Module):
    def __init__(self, dropout_ratio=0.3, exp_rate=2.25, num_sample=20, adapter_dim=8, r_ssp=0.3):
        super(MMDG2, self).__init__()
        self.num_encoders = 5
        dim = 768

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter1 = []
        self.adapter2 = []
        self.ca1 = []
        self.ca2 = []
        self.ln1 = []
        self.ln2 = []

        for i in range(self.num_encoders):
            self.adapter1.append(ConvAdapter(adapter_dim=adapter_dim))
            self.adapter2.append(ConvAdapter(adapter_dim=adapter_dim))
            self.ca1.append(My_MHA_UC(dim=dim))
            self.ca2.append(My_MHA_UC(dim=dim))
            self.ln1.append(nn.LayerNorm(dim))
            self.ln2.append(nn.LayerNorm(dim))

        self.adapter1 = nn.Sequential(*self.adapter1)
        self.adapter2 = nn.Sequential(*self.adapter2)
        self.ca1 = nn.Sequential(*self.ca1)
        self.ca2 = nn.Sequential(*self.ca2)
        self.ln1 = nn.Sequential(*self.ln1)
        self.ln2 = nn.Sequential(*self.ln2)

        self.out = None

    def forward(self, x1, x2, domain=None):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        for i in range(1, len(self.ViT_Encoder) - 1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            y1 = self.ViT_Encoder[i].ln_2(x1_a)
            y2 = self.ViT_Encoder[i].ln_2(x2_a)

            y1_m = self.adapter1[i - 1](y1)
            y2_m = self.adapter2[i - 1](y2)
            y1_k = self.ca1[i - 1](y2_m, y1_m, y1_m) + y1_m
            y2_k = self.ca1[i - 1](y1_m, y2_m, y2_m) + y2_m

            y1_k = self.ln1[i - 1](y1_k)
            y2_k = self.ln2[i - 1](y2_k)

            proj1 = self.ViT_Encoder[i].mlp(y1) + x1_a + y1_k
            proj2 = self.ViT_Encoder[i].mlp(y2) + x2_a + y2_k

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class MMDG3(nn.Module):
    def __init__(self, dropout_ratio=0.3, exp_rate=2.25, num_sample=20, adapter_dim=8, r_ssp=0.3):
        super(MMDG3, self).__init__()
        self.num_encoders = 5
        dim = 768

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter1 = []
        self.adapter2 = []
        self.ca1 = []
        self.ca2 = []
        self.ln1 = []
        self.ln2 = []

        for i in range(self.num_encoders):
            self.adapter1.append(ConvAdapter(adapter_dim=adapter_dim))
            self.adapter2.append(ConvAdapter(adapter_dim=adapter_dim))
            self.ca1.append(My_MHA_UC(dim=dim))
            self.ca2.append(My_MHA_UC(dim=dim))
            self.ln1.append(nn.LayerNorm(dim))
            self.ln2.append(nn.LayerNorm(dim))

        self.adapter1 = nn.Sequential(*self.adapter1)
        self.adapter2 = nn.Sequential(*self.adapter2)
        self.ca1 = nn.Sequential(*self.ca1)
        self.ca2 = nn.Sequential(*self.ca2)
        self.ln1 = nn.Sequential(*self.ln1)
        self.ln2 = nn.Sequential(*self.ln2)

        self.out = None

    def forward(self, x1, x2, domain=None):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        for i in range(1, len(self.ViT_Encoder) - 1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            y1 = self.ViT_Encoder[i].ln_2(x1_a)
            y2 = self.ViT_Encoder[i].ln_2(x2_a)

            y1_m = self.ca1[i - 1](y2, y1, y1) + x1_a
            y2_m = self.ca1[i - 1](y1, y2, y2) + x2_a

            y1_k = self.ln1[i - 1](y1_m)
            y2_k = self.ln2[i - 1](y2_m)

            y1_k = self.adapter1[i - 1](y1_k)
            y2_k = self.adapter2[i - 1](y2_k)

            proj1 = self.ViT_Encoder[i].mlp(y1) + x1_a + y1_k
            proj2 = self.ViT_Encoder[i].mlp(y2) + x2_a + y2_k

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class STB(nn.Module):
    def __init__(self,dim,num_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim,)
        self.MSA = nn.MultiheadAttention(dim, num_head, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(dim,)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        x1 = self.ln1(x)
        x1 = self.MSA(x1, x1, x1)[0] + x
        x2 = self.ln2(x1)
        x2 = self.mlp(x2) + x1
        return x2

class MMA(nn.Module):
    def __init__(
            self,
            dim=768,
            num_heads=8,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.xq = nn.Linear(dim, dim, bias=False)
        self.xk = nn.Linear(dim, dim, bias=False)
        self.yq = nn.Linear(dim, dim, bias=False)
        self.yk = nn.Linear(dim, dim, bias=False)
        self.yv = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y):  #y가 메인
        B, N, C = x.shape
        x_patch = x[:, 1:, :]  #B, 14*14, 768
        x_cls = x[:, :1, :]  #B, 1, 768
        y_patch = x[:, 1:, :]
        y_cls = x[:, :1, :]

        x_q = self.xq(x_cls).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2) #B, 8, 1, 96
        x_k = self.xk(x_patch).reshape(B, 14*14, self.num_heads, self.head_dim).transpose(1, 2) #B, 8, 196, 96
        x_q = x_q * self.scale
        attn_x = x_q @ x_k.transpose(-2, -1)  # B, 8, 1, 14*14
        attn_x = torch.where(attn_x >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

        y_q = self.yq(y_cls).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        y_k = self.yk(y_patch).reshape(B, 14*14, self.num_heads, self.head_dim).transpose(1, 2)
        y_q = y_q * self.scale
        attn_y = y_q @ y_k.transpose(-2, -1)  # B, 8, 1, 14*14
        attn_y = torch.where(attn_y >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

        attn = attn_x + attn_y
        attn = torch.where(attn > 0, attn_y, torch.tensor(-10**8))
        attn = attn.softmax(dim=-1)

        y_v = self.yv(y_patch).reshape(B, 14*14, self.num_heads, self.head_dim).transpose(1, 2)
        y_cls2 = attn @ y_v  #B, 8, 1, 96

        y_cls2 = y_cls2.transpose(1, 2).reshape(B, 1, C)  #B, 1, 768
        y_cls2 = self.proj(y_cls2) + y_cls
        y = torch.cat([y_cls2, y_patch], dim=1)

        return y

class MFA(nn.Module):
    def __init__(
            self,
            dim=768,
            num_heads=8,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.xk = nn.Linear(dim, dim, bias=False)
        self.xv = nn.Linear(dim, dim, bias=False)
        self.yq = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y):  #y가 메인
        B, N, C = x.shape
        x_patch = x[:, 1:, :]  #B, 14*14, 768
        x_cls = x[:, :1, :]  #B, 1, 768
        y_patch = x[:, 1:, :]
        y_cls = x[:, :1, :]

        x_k = self.xk(x_patch).reshape(B, 14*14, self.num_heads, self.head_dim).transpose(1, 2) #B, 8, 196, 96
        x_v = self.xv(x_patch).reshape(B, 14*14, self.num_heads, self.head_dim).transpose(1, 2)
        y_q = self.yq(y_cls).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        y_q = y_q * self.scale
        attn = y_q @ x_k.transpose(-2, -1)  # B, 8, 1, 14*14
        attn = attn.softmax(dim=-1)
        y_cls2 = attn @ x_v  #B, 8, 1, 96

        y_cls2 = y_cls2.transpose(1, 2).reshape(B, 1, C)  #B, 1, 768
        y_cls2 = self.proj(y_cls2) + y_cls
        y = torch.cat([y_cls2, y_patch], dim=1)

        return y

class FMViT(nn.Module):
    def __init__(self):
        super(FMViT, self).__init__()
        self.num_encoders = 3
        dim = 768
        self.ln1 = nn.LayerNorm(768)
        self.ln2 = nn.LayerNorm(768)

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False
        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        layer_list.append(vit.ln)

        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.stb1 = nn.Sequential(*[STB(dim=768, num_head=12) for _ in range(2)])
        self.stb2 = nn.Sequential(*[STB(dim=768, num_head=12) for _ in range(2)])
        self.stb3 = nn.Sequential(*[STB(dim=768, num_head=12) for _ in range(4)])
        self.stb_total = [self.stb1, self.stb2, self.stb3]
        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter1_1 = []
        self.adapter1_2 = []
        self.adapter2_1 = []
        self.adapter2_2 = []
        for i in range(self.num_encoders):
            self.adapter1_1.append(MMA())
            self.adapter1_2.append(MFA())
            self.adapter2_1.append(MMA())
            self.adapter2_2.append(MFA())
        self.adapter1_1 = nn.Sequential(*self.adapter1_1)
        self.adapter1_2 = nn.Sequential(*self.adapter1_2)
        self.adapter2_1 = nn.Sequential(*self.adapter2_1)
        self.adapter2_2 = nn.Sequential(*self.adapter2_2)
        self.out = None

    def forward(self, x1, x2):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        for i in range(1, len(self.ViT_Encoder) - 1):
            y1_m = self.stb_total[i-1](proj1)
            y2_m = self.stb_total[i-1](proj2)

            y1_c = self.ln1(y1_m)
            y2_c = self.ln1(y2_m)
            y1_d = self.adapter1_1[i - 1](y2_c, y1_c) + y1_m
            y2_d = self.adapter2_1[i - 1](y1_c, y2_c) + y2_m

            y2_e = self.ln2(y2_d)
            y1_e = self.ln2(y1_d)
            proj1 = self.adapter1_2[i - 1](y2_c, y1_e) + y1_d
            proj2 = self.adapter2_2[i - 1](y1_c, y2_e) + y2_d

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        feat_1 = self.fc_one(logits1)
        feat_2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": feat_1,
            "feat_2": feat_2
        }
        return self.out

class MDA(nn.Module):
    def __init__(
            self,
            dim=768,
            num_heads=8,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.mq = nn.Linear(dim, dim, bias=False)
        self.mk = nn.Linear(dim, dim, bias=False)
        self.cq = nn.Linear(dim, dim, bias=False)
        self.ck = nn.Linear(dim, dim, bias=False)
        self.yv = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, y):
        B, N, C = y.shape
        y_patch = y[:, 2:, :]  #B, 14*14, 768
        y_cls = y[:, :1, :]  #B, 1, 768
        y_mod = y[:, 1:2, :]  #B, 1, 768

        y_mq = self.mq(y_mod).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2) #B, 8, 1, 96
        y_mk = self.mk(y_patch).reshape(B, 14*14, self.num_heads, self.head_dim).transpose(1, 2) #B, 8, 196, 96
        y_mq = y_mq * self.scale
        attn_m = y_mq @ y_mk.transpose(-2, -1)  # B, 8, 1, 14*14
        #attn_m = attn_m.softmax(dim=-1)
        attn_m = torch.where(attn_m >= 0.8, torch.tensor(1.0), torch.tensor(0.0))

        y_cq = self.cq(y_cls).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # B, 8, 1, 96
        y_ck = self.ck(y_patch).reshape(B, 14 * 14, self.num_heads, self.head_dim).transpose(1, 2)  # B, 8, 196, 96
        y_cq = y_cq * self.scale
        attn_c = y_cq @ y_ck.transpose(-2, -1)  # B, 8, 1, 14*14

        attn = torch.where(attn_m == 0, attn_c, torch.tensor(-10**8))
        attn = attn.softmax(dim=-1)

        y_v = self.yv(y_patch).reshape(B, 14*14, self.num_heads, self.head_dim).transpose(1, 2)
        y_cls2 = attn @ y_v  #B, 8, 1, 96

        y_cls2 = y_cls2.transpose(1, 2).reshape(B, 1, C)  #B, 1, 768
        y_cls2 = self.proj(y_cls2) + y_cls
        y = torch.cat([y_cls2, y_mod, y_patch], dim=1)

        return y

class CMA(nn.Module):
    def __init__(
            self,
            dim=768,
            num_heads=8,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.xk = nn.Linear(dim, dim, bias=False)
        self.xv = nn.Linear(dim, dim, bias=False)
        self.yq = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y):  #y가 메인
        B, N, C = x.shape

        x_k = self.xk(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) #B, 8, 196, 96
        x_v = self.xv(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        y_q = self.yq(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        y_q = y_q * self.scale
        attn = y_q @ x_k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        y_cls2 = attn @ x_v

        y_cls2 = y_cls2.transpose(1, 2).reshape(B, 1, C)
        y = self.proj(y_cls2)

        return y

class MAViT(nn.Module):
    def __init__(self):
        super(MAViT, self).__init__()
        self.num_encoders = 5
        self.ln = nn.LayerNorm(768)
        dim = 768

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False
        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)
        self.modal_token1 = copy.deepcopy(self.class_token1)
        self.modal_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        seq_length = 14 * 14 + 1 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        layer_list.append(vit.ln)

        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_cls = nn.Linear(dim, 2)
        self.fc_mod = nn.Linear(dim, 2)

        self.adapter1 = []
        self.adapter2 = []
        #self.adapter3 = []
        for i in range(self.num_encoders):
            self.adapter1.append(MDA())
            self.adapter2.append(MDA())
            #self.adapter3.append(CMA())
        self.adapter1 = nn.Sequential(*self.adapter1)
        self.adapter2 = nn.Sequential(*self.adapter2)
        #self.adapter3 = nn.Sequential(*self.adapter3)
        self.out = None
        self.MATB_mlp1 = nn.Sequential(*[nn.Linear(768,768)]*(len(self.ViT_Encoder)-2))
        self.MATB_mlp2 = nn.Sequential(*[nn.Linear(768,768)]*(len(self.ViT_Encoder)-2))


    def forward(self, x1, x2):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.modal_token1.expand(b, -1, -1), x1), dim=1)
        x2 = torch.cat((self.modal_token2.expand(b, -1, -1), x2), dim=1)
        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)
        proj3 = None

        for i in range(1, len(self.ViT_Encoder) - 1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)

            ca1, _ = self.ViT_Encoder[i].self_attention(x1, x2, x2, need_weights=False)
            ca2, _ = self.ViT_Encoder[i].self_attention(x2, x1, x1, need_weights=False)
            ca = torch.cat((ca1, ca2), dim=1)
            if proj3 is not None:
                ca = ca + proj3
            ca_e = self.ln(ca)
            proj3 = self.ViT_Encoder[i].mlp(ca_e) + ca

            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False) # B 1 N , B 1 N, B hw N

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            y1 = self.ViT_Encoder[i].ln_2(x1_a)
            y2 = self.ViT_Encoder[i].ln_2(x2_a)
            y1_m = self.ViT_Encoder[i].mlp(y1) + x1_a
            y2_m = self.ViT_Encoder[i].mlp(y2) + x2_a

            y1_c = self.ln(y1_m)
            y2_c = self.ln(y2_m)
            y1_d = self.adapter1[i - 1](y1_c) + y1_m
            y2_d = self.adapter2[i - 1](y2_c) + y2_m

            y1_e = self.ln(y1_d)
            y2_e = self.ln(y2_d)

            proj1 = self.MATB_mlp1[i-1](y1_e) + y1_d
            proj2 = self.MATB_mlp1[i - 1](y2_e) + y2_d
            #proj2 = self.MATB_mlp2[i-1](y2_e) + y2_d

            #proj1 = self.ViT_Encoder[i].mlp(y1_e) + y1_d
            #proj2 = self.ViT_Encoder[i].mlp(y2_e) + y2_d

        proj1 = self.ViT_Encoder[-1](proj1) # nn.Layernorm
        proj2 = self.ViT_Encoder[-1](proj2)
        proj3 = self.ViT_Encoder[-1](proj3)

        logits1c = proj1[:, 0]
        logits2c = proj2[:, 0]
        logits1m = proj1[:, 1]
        logits2m = proj2[:, 1]
        logits3 = proj3[:, 0]
        c1 = self.fc_cls(logits1c)
        c2 = self.fc_cls(logits2c)
        m1 = self.fc_mod(logits1m)
        m2 = self.fc_mod(logits2m)
        c3 = self.fc_cls(logits3)
        self.out = {"1c": c1, "2c": c2, "1m": m1, "2m": m2, "3c": c3}
        return self.out

class ViT(nn.Module):
    '''
    trainable params : 31M
    '''
    def __init__(self, hidden_dim=768):
        super(ViT, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.ln_set = nn.ModuleList([nn.LayerNorm(768) for _ in range(self.num_encoders)])
        #self.ln1 = nn.LayerNorm(768)

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter_1_2_2 = []
        self.adapter_2_1_2 = []

        for i in range(self.num_encoders):
            self.adapter_1_2_2.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_2_1_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_1_2_2 = nn.Sequential(*self.adapter_1_2_2)
        self.adapter_2_1_2 = nn.Sequential(*self.adapter_2_1_2)

        self.out = None

    def forward(self, x1, x2):
        #original1 = x1.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
        #original2 = x2.squeeze(0).permute(1, 2, 0).numpy()

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        attn_feats = defaultdict(list)
        mlp_feats = defaultdict(list)
        for i in range(1, len(self.ViT_Encoder)-1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            y1 = x1_a + self.adapter_1_2_2[i - 1](x2_b, x1_b, x1_b)  #ln 이전 더하기
            y2 = x2_a + self.adapter_2_1_2[i - 1](x1_b, x2_b, x2_b)

            # attn_feats['rgb'].append(y1)
            # attn_feats['ir'].append(y2)

            y1_a = self.ln_set[i-1](y1)
            y2_a = self.ln_set[i-1](y2)

            proj1 = self.ViT_Encoder[i].mlp(y1_a) + y1
            proj2 = self.ViT_Encoder[i].mlp(y2_a) + y2

            # mlp_feats['rgb'].append(proj1)
            # mlp_feats['ir'].append(proj2)
        attn_feats['rgb'].append(y1)

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2,
            "attn_feats":attn_feats,
            "mlp_feats":mlp_feats
        }
        return self.out

class ViT2(nn.Module):
    def __init__(self, hidden_dim=768):
        super(ViT2, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.ln_set = [nn.LayerNorm(768) for _ in range(self.num_encoders)]
        #self.ln1 = nn.LayerNorm(768)

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter_1_2_2 = []
        self.adapter_2_1_2 = []

        for i in range(self.num_encoders):
            self.adapter_1_2_2.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_2_1_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_1_2_2 = nn.Sequential(*self.adapter_1_2_2)
        self.adapter_2_1_2 = nn.Sequential(*self.adapter_2_1_2)

        self.out = None

    def forward(self, x1, x2):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        for i in range(1, len(self.ViT_Encoder)-1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            y1 = x1_a + self.adapter_1_2_2[i - 1](x2_b, x1_b, x1_b)  #ln 이전 더하기
            y2 = x2_a + self.adapter_2_1_2[i - 1](x1_b, x2_b, x2_b)

            y1_a = self.ln_set[i-1](y1)
            y2_a = self.ln_set[i-1](y2)

            proj1 = self.ViT_Encoder[i].mlp(y1_a) + y1
            proj2 = self.ViT_Encoder[i].mlp(y2_a) + y2

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class ViT3(nn.Module):
    def __init__(self, hidden_dim=768):
        super(ViT3, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.ln_set = [nn.LayerNorm(768) for _ in range(self.num_encoders)]
        #self.ln1 = nn.LayerNorm(768)

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)
        self.conv_proj3 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)
        self.class_token3 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding3 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter_1_2_2 = []
        self.adapter_2_1_2 = []

        for i in range(self.num_encoders):
            self.adapter_1_2_2.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_2_1_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_1_2_2 = nn.Sequential(*self.adapter_1_2_2)
        self.adapter_2_1_2 = nn.Sequential(*self.adapter_2_1_2)

        self.out = None

    def forward(self, x1, x2, x3):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)
        x2 = x2.flatten(2).transpose(1, 2)

        x3 = self.conv_proj3(x3)
        x3 = x3.flatten(2).transpose(1, 2)

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)
        x3 = torch.cat((self.class_token3.expand(b, -1, -1), x3), dim=1)

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2
        proj3 = x3 + self.pos_embedding3

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)
        proj3 = self.ViT_Encoder[0](proj3)

        for i in range(1, len(self.ViT_Encoder)-1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x3 = self.ViT_Encoder[i].ln_1(proj3)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)
            x3_c, _ = self.ViT_Encoder[i].self_attention(x3, x3, x3, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c
            x3_a = proj3 + x3_c

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)
            x3_b = self.ViT_Encoder[i].ln_2(x3_a)

            y2 = x2_a + self.adapter_1_2_2[i - 1](x1_b, x2_b, x2_b) #ir+rgb
            y3 = x3_a + self.adapter_2_1_2[i - 1](x1_b, x3_b, x3_b) #depth+rgb

            y2_a = self.ln_set[i-1](y2)
            y3_a = self.ln_set[i-1](y3)

            proj1 = self.ViT_Encoder[i].mlp(x1_b) + x1_a
            proj2 = self.ViT_Encoder[i].mlp(y2_a) + y2
            proj3 = self.ViT_Encoder[i].mlp(y3_a) + y3

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)
        proj3 = self.ViT_Encoder[-1](proj3)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]
        logits3 = proj3[:, 0]
        logits_all = torch.cat([logits1, logits2, logits3], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        logits3 = self.fc_one(logits3)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2,
            "feat_3": logits3
        }
        return self.out

class ViT_prompt(nn.Module):
    def __init__(self, hidden_dim=768):
        super(ViT_prompt, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.prompt_length = 2

        #self.ln1 = nn.LayerNorm(dim=768)
        #self.ln2 = nn.LayerNorm(dim=768)
        self.prompts1 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 768)) for _ in range(self.num_encoders)])
        for a in self.prompts1:
            nn.init.normal_(a, std=0.02)
        self.prompts2 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 768)) for _ in range(self.num_encoders)])
        for a in self.prompts2:
            nn.init.normal_(a, std=0.02)

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list) #dropout + num_encoders + ln

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter_0_1 = []
        self.adapter_0_2 = []

        for _ in range(self.num_encoders):
            self.adapter_0_1.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_0_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_0_1 = nn.Sequential(*self.adapter_0_1)
        self.adapter_0_2 = nn.Sequential(*self.adapter_0_2)

        self.out = None

    def forward(self, x1, x2):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        for i in range(1, len(self.ViT_Encoder)-1):
            q1 = self.adapter_0_1[i-1](self.prompts1[i-1].expand(b, -1, -1), proj1, proj1)
            q2 = self.adapter_0_2[i-1](self.prompts2[i-1].expand(b, -1, -1), proj2, proj2)
            proj1 = torch.cat((proj1, q2), dim=1)
            proj2 = torch.cat((proj2, q1), dim=1)

            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            #prompt 제거
            x1_a = x1_a[:, :-self.prompt_length, :]
            x2_a = x2_a[:, :-self.prompt_length, :]

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            proj1 = self.ViT_Encoder[i].mlp(x1_b) + x1_a
            proj2 = self.ViT_Encoder[i].mlp(x2_b) + x2_a

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class ViT_text(nn.Module):
    def __init__(self, hidden_dim=768):
        super(ViT_text, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.prompt_length = 2

        #self.ln1 = nn.LayerNorm(dim=768)
        #self.ln2 = nn.LayerNorm(dim=768)
        self.prompts1 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 512)) for _ in range(self.num_encoders)])
        for a in self.prompts1:
            nn.init.normal_(a, std=0.02)
        self.prompts2 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 512)) for _ in range(self.num_encoders)])
        for a in self.prompts2:
            nn.init.normal_(a, std=0.02)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list) #dropout + (num_encoders-1) + ln

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)
        self.text_lin1 = nn.Linear(512, dim)
        self.text_lin2 = nn.Linear(512, dim)

        self.adapter_0_1 = []
        self.adapter_0_2 = []

        for _ in range(self.num_encoders):
            self.adapter_0_1.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_0_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_0_1 = nn.Sequential(*self.adapter_0_1)
        self.adapter_0_2 = nn.Sequential(*self.adapter_0_2)

        self.out = None

        self.text_labels = ["a photo of a real face", "a photo of a paper attack", "a photo of a display attack", "a photo of a mask attack", "a photo of a"]
        self.text_types = ["real face", "paper attack", "display attack", "mask attack", "None"]
        self.text_token = [clip.tokenize(t).to(self.device) for t in self.text_labels]
        self.text_embed = [self.clip_model.token_embedding(t).type(self.clip_model.dtype) for t in self.text_token]

    def forward(self, x1, x2, text=None):

        b, c, fh, fw = x1.shape

        text_prompts = {label: {"rgb": [], "ir": []} for label in self.text_types}
        if text is None:
            text = ["None"] * b

        for label, embed, token in zip(self.text_types, self.text_embed, self.text_token):
            for layer_idx in range(self.num_encoders):
                rgb_prompt = self.text_lin1(self.clip_model.encode_text(embed, self.prompts1[layer_idx], token))  #(1, 77+2, 512) > (1, 768)
                ir_prompt = self.text_lin2(self.clip_model.encode_text(embed, self.prompts2[layer_idx], token))

                text_prompts[label]["rgb"].append(rgb_prompt)
                text_prompts[label]["ir"].append(ir_prompt)

        #if text is None:
            #text = ["a photo of a"] * b
        #else:
            #text = ["a photo of a " + t for t in text]  #(b, 77)
        #text = self.clip_model.encode_text(clip.tokenize(text).to(self.device))  #(b, 512)
        #text = self.text_lin(text)

        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        for i in range(1, len(self.ViT_Encoder)-1):
            prompt_list1 = []
            prompt_list2 = []
            for t in text:
                prompt1 = text_prompts[t]["rgb"][i-1]  #(1, 768)
                prompt_list1.append(prompt1)
                prompt2 = text_prompts[t]["ir"][i-1]
                prompt_list2.append(prompt2)
            prompt_batch1 = torch.cat(prompt_list1, dim=0).unsqueeze(1)  #(b, 1, 768)
            prompt_batch2 = torch.cat(prompt_list2, dim=0).unsqueeze(1)

            #q1 = self.adapter_0_1[i-1](torch.cat((self.prompts1[i-1].expand(b, -1, -1), text.unsqueeze(1)), dim = 1), proj1, proj1)
            #q2 = self.adapter_0_2[i-1](torch.cat((self.prompts2[i-1].expand(b, -1, -1), text.unsqueeze(1)), dim = 1), proj2, proj2)
            q1 = self.adapter_0_1[i - 1](prompt_batch1, proj1, proj1)
            q2 = self.adapter_0_2[i - 1](prompt_batch2, proj2, proj2)

            proj1 = torch.cat((proj1, q2), dim=1)
            proj2 = torch.cat((proj2, q1), dim=1)

            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            #prompt 제거
            x1_a = x1_a[:, :-1, :]
            x2_a = x2_a[:, :-1, :]

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            proj1 = self.ViT_Encoder[i].mlp(x1_b) + x1_a
            proj2 = self.ViT_Encoder[i].mlp(x2_b) + x2_a

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class ViT_text2(nn.Module):
    def __init__(self, hidden_dim=768):
        super(ViT_text2, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.prompt_length = 2

        #self.ln1 = nn.LayerNorm(dim=768)
        #self.ln2 = nn.LayerNorm(dim=768)
        self.prompts1 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 512)) for _ in range(self.num_encoders)])
        for a in self.prompts1:
            nn.init.normal_(a, std=0.02)
        self.prompts2 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 512)) for _ in range(self.num_encoders)])
        for a in self.prompts2:
            nn.init.normal_(a, std=0.02)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list) #dropout + (num_encoders-1) + ln

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)
        self.text_lin1 = nn.Sequential(*[nn.Linear(512, dim) for _ in range(self.num_encoders)])
        self.text_lin2 = nn.Sequential(*[nn.Linear(512, dim) for _ in range(self.num_encoders)])

        self.adapter_0_1 = []
        self.adapter_0_2 = []

        for _ in range(self.num_encoders):
            self.adapter_0_1.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_0_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_0_1 = nn.Sequential(*self.adapter_0_1)
        self.adapter_0_2 = nn.Sequential(*self.adapter_0_2)

        self.out = None

        self.text_labels = ["a photo of a real face", "a photo of a paper attack", "a photo of a display attack", "a photo of a mask attack", "a photo of a"]
        self.text_types = ["real face", "paper attack", "display attack", "mask attack", "None"]
        self.text_token = [clip.tokenize(t).to(self.device) for t in self.text_labels]
        self.text_embed = [self.clip_model.token_embedding(t).type(self.clip_model.dtype) for t in self.text_token]

    def forward(self, x1, x2, text=None):

        b, c, fh, fw = x1.shape

        text_prompts = {label: {"rgb": [], "ir": []} for label in self.text_types}
        if text is None:
            text = ["None"] * b

        for label, embed, token in zip(self.text_types, self.text_embed, self.text_token):
            rgb_feats = self.clip_model.encode_text(embed, self.prompts1, token)[1]
            ir_feats = self.clip_model.encode_text(embed, self.prompts2, token)[1]
            for layer_idx in range(self.num_encoders):
                rgb_prompt = self.text_lin1[layer_idx](rgb_feats[layer_idx][torch.arange(rgb_feats[layer_idx].shape[0]), token.argmax(dim=-1)+2])  #(1, 77+2, 512) > (1, 768)
                ir_prompt = self.text_lin2[layer_idx](ir_feats[layer_idx][torch.arange(ir_feats[layer_idx].shape[0]), token.argmax(dim=-1)+2])
                text_prompts[label]["rgb"].append(rgb_prompt)
                text_prompts[label]["ir"].append(ir_prompt)

        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        for i in range(1, len(self.ViT_Encoder)-1):
            prompt_list1 = []
            prompt_list2 = []
            for t in text:
                prompt1 = text_prompts[t]["rgb"][i-1]  #(1, 768)
                prompt_list1.append(prompt1)
                prompt2 = text_prompts[t]["ir"][i-1]
                prompt_list2.append(prompt2)
            prompt_batch1 = torch.cat(prompt_list1, dim=0).unsqueeze(1)  #(b, 1, 768)
            prompt_batch2 = torch.cat(prompt_list2, dim=0).unsqueeze(1)

            #q1 = self.adapter_0_1[i-1](torch.cat((self.prompts1[i-1].expand(b, -1, -1), text.unsqueeze(1)), dim = 1), proj1, proj1)
            #q2 = self.adapter_0_2[i-1](torch.cat((self.prompts2[i-1].expand(b, -1, -1), text.unsqueeze(1)), dim = 1), proj2, proj2)
            q1 = self.adapter_0_1[i - 1](prompt_batch1, proj1, proj1)
            q2 = self.adapter_0_2[i - 1](prompt_batch2, proj2, proj2)

            proj1 = torch.cat((proj1, q2), dim=1)
            proj2 = torch.cat((proj2, q1), dim=1)

            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            #prompt 제거
            x1_a = x1_a[:, :-1, :]
            x2_a = x2_a[:, :-1, :]

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            proj1 = self.ViT_Encoder[i].mlp(x1_b) + x1_a
            proj2 = self.ViT_Encoder[i].mlp(x2_b) + x2_a

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class ViT_text3(nn.Module):
    def __init__(self, hidden_dim=768):
        super(ViT_text3, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.prompt_length = 2

        #self.ln1 = nn.LayerNorm(dim=768)
        #self.ln2 = nn.LayerNorm(dim=768)
        self.prompts1 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 512)) for _ in range(self.num_encoders)]) # [2 512] x 4
        for a in self.prompts1:
            nn.init.normal_(a, std=0.02)
        self.prompts2 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 512)) for _ in range(self.num_encoders)])
        for a in self.prompts2:
            nn.init.normal_(a, std=0.02)

        # CLIP 모델 생성
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model = build_model(self.clip_model.state_dict()).to(self.device) # CLIP_feats()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ViT 모델 생성
        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False


        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]  #layer0
        for i in range(self.num_encoders):   #layer1~4, layer5=requires_grad
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)   #layer6

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list) #dropout + (num_encoders) + ln

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)
        self.text_lin1 = nn.Sequential(*[nn.Linear(512, dim) for _ in range(self.num_encoders)])
        self.text_lin2 = nn.Sequential(*[nn.Linear(512, dim) for _ in range(self.num_encoders)])

        self.text_final = nn.Linear(512, dim)
        self.text_fake = nn.Linear(dim*3, dim)

        self.adapter_0_1 = []
        self.adapter_0_2 = []

        for _ in range(self.num_encoders):
            self.adapter_0_1.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_0_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_0_1 = nn.Sequential(*self.adapter_0_1)
        self.adapter_0_2 = nn.Sequential(*self.adapter_0_2)

        self.out = None

        self.text_labels = ["a photo of a real face", "a photo of a paper attack", "a photo of a display attack", "a photo of a mask attack"]
        #self.text_labels = ["a photo of a real face", "a photo of a fake face"]
        #self.text_labels = ["a photo of a real face", "a photo of a paper, display, mask face"]
        #self.text_labels = ["a photo of a real face", "a photo of a paper face", "a photo of a display face", "a photo of a mask face"]

        self.text_token = clip.tokenize(self.text_labels).to(self.device)  # (4, 77)

    def forward(self, x1, x2):

        b, c, fh, fw = x1.shape

        self.rgb_feats = self.clip_model.encode_text(self.text_token, self.prompts1)  # [4,d] x 4
        self.ir_feats = self.clip_model.encode_text(self.text_token, self.prompts2)  # [4,d] x 4

        for idx in range(self.num_encoders):
            self.rgb_feats[idx] = self.text_lin1[idx](self.rgb_feats[idx])
            self.ir_feats[idx] = self.text_lin2[idx](self.ir_feats[idx])

        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj1(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        for i in range(1, len(self.ViT_Encoder)-1):  #i = 1~5
            q1 = self.adapter_0_1[i - 1](self.rgb_feats[i-1].unsqueeze(0).repeat(proj1.shape[0], 1, 1), proj1, proj1)  #b,4,768
            q2 = self.adapter_0_2[i - 1](self.ir_feats[i-1].unsqueeze(0).repeat(proj2.shape[0], 1, 1), proj2, proj2)

            proj1 = torch.cat((proj1, q2), dim=1)  #b,197+4,768
            proj2 = torch.cat((proj2, q1), dim=1)

            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            #prompt 제거
            x1_a = x1_a[:, :-2, :]  #-4
            x2_a = x2_a[:, :-2, :]

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            proj1 = self.ViT_Encoder[i].mlp(x1_b) + x1_a
            proj2 = self.ViT_Encoder[i].mlp(x2_b) + x2_a

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)
        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d

        #eos = self.text_final(self.rgb_feats[-1])  #4,768
        #weights = torch.cat([eos[0].unsqueeze(0), eos[1:,:].mean(dim=0, keepdim=True)], dim=0)  # (2, 768)

        weights = self.text_final(self.rgb_feats[-1])  #2, 768  #4, 768

        real = weights[0, :].unsqueeze(0) #1, 768
        fake = self.text_fake(weights[1:, :].view(-1)).view(1, 768)
        weights = torch.cat([real, fake], dim=0)

        #cls_feat @ weights.T → (b, 768) @ (768, 2) = (b, 2)
        logits1 = logits1 @ weights.T  # (b, 2)
        logits2 = logits2 @ weights.T


        self.out = {
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class CDC(nn.Module):
    def __init__(self, adapter_dim=8, theta=0.5, use_cdc=True, dropout_rate=0.2, hidden_dim=768):
        super(CDC, self).__init__()
        if use_cdc:
            self.adapter_conv = Conv2d_cd(in_channels=adapter_dim, out_channels=adapter_dim,
                                          kernel_size=3, stride=1, padding=1, theta=theta)
        else:
            self.adapter_conv = nn.Conv2d(in_channels=adapter_dim, out_channels=adapter_dim,
                                          kernel_size=3, stride=1, padding=1)
        self.cross_attention = My_MHA_UC(dim=adapter_dim)
        self.ln_before = nn.LayerNorm(adapter_dim)
        # nn.init.xavier_uniform_(self.adapter_conv.conv.weight)  # CDC xavier初始化
        # nn.init.zeros_(self.adapter_conv.conv.bias)

        self.adapter_down_1 = nn.Linear(hidden_dim, adapter_dim)  # equivalent to 1 * 1 Conv
        self.adapter_down_2 = nn.Linear(hidden_dim, adapter_dim)  # equivalent to 1 * 1 Conv

        self.adapter_up = nn.Linear(adapter_dim, hidden_dim)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down_1.weight)
        nn.init.zeros_(self.adapter_down_1.bias)
        nn.init.xavier_uniform_(self.adapter_down_2.weight)
        nn.init.zeros_(self.adapter_down_2.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.dim = adapter_dim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down_1(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:(1 + 14 * 14)]
        x_patch = x_patch.reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up

class ViT_prompt_CDC(nn.Module):
    def __init__(self, hidden_dim=768):
        super(ViT_prompt_CDC, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.prompt_length = 2

        #self.ln1 = nn.LayerNorm(dim=768)
        #self.ln2 = nn.LayerNorm(dim=768)
        self.prompts1 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 768)) for _ in range(self.num_encoders)])
        for a in self.prompts1:
            nn.init.normal_(a, std=0.02)
        self.prompts2 = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 768)) for _ in range(self.num_encoders)])
        for a in self.prompts2:
            nn.init.normal_(a, std=0.02)

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list) #dropout + num_encoders + ln

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter_0_1 = []
        self.adapter_0_2 = []
        self.cdc1 = []
        self.cdc2 = []

        for _ in range(self.num_encoders):
            self.adapter_0_1.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_0_2.append(My_MHA_UC(dim=hidden_dim))
            self.cdc1.append(CDC())
            self.cdc2.append(CDC())

        self.adapter_0_1 = nn.Sequential(*self.adapter_0_1)
        self.adapter_0_2 = nn.Sequential(*self.adapter_0_2)
        self.cdc1 = nn.Sequential(*self.cdc1)
        self.cdc2 = nn.Sequential(*self.cdc2)

        self.out = None

    def forward(self, x1, x2):

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        for i in range(1, len(self.ViT_Encoder)-1):
            q1 = self.adapter_0_1[i-1](self.prompts1[i-1].expand(b, -1, -1), proj1, proj1)
            q2 = self.adapter_0_1[i-1](self.prompts2[i-1].expand(b, -1, -1), proj2, proj2)
            proj1 = torch.cat((proj1, q2), dim=1)
            proj2 = torch.cat((proj2, q1), dim=1)

            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            #prompt 제거
            x1_a = x1_a[:, :-self.prompt_length, :]
            x2_a = x2_a[:, :-self.prompt_length, :]

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            #CDC
            z1 = self.cdc1[i - 1](x1_b)
            z2 = self.cdc2[i - 1](x2_b)

            proj1 = self.ViT_Encoder[i].mlp(x1_b) + x1_a + z1
            proj2 = self.ViT_Encoder[i].mlp(x2_b) + x2_a + z2

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class Diff_attention(nn.Module):
    def __init__(self, hidden_dim=768):
        super(Diff_attention, self).__init__()
        self.num_encoders = 5
        dim = 768
        self.ln_set = nn.ModuleList([nn.LayerNorm(768) for _ in range(self.num_encoders)])
        #self.ln1 = nn.LayerNorm(768)

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        #for p in self.conv_proj1.parameters():
        #    p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            #if i < (self.num_encoders - 1):
                #for p in vit.layers[i].parameters():
                #    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter_1_2_2 = []
        self.adapter_2_1_2 = []

        for i in range(self.num_encoders):
            self.adapter_1_2_2.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_2_1_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_1_2_2 = nn.Sequential(*self.adapter_1_2_2)
        self.adapter_2_1_2 = nn.Sequential(*self.adapter_2_1_2)

        self.out = None

    def forward(self, x1, x2):
        #original1 = x1.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
        #original2 = x2.squeeze(0).permute(1, 2, 0).numpy()

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        ''' feature map 시각화 '''
        attn_feats = defaultdict(list)
        mlp_feats = defaultdict(list)
        for i in range(1, len(self.ViT_Encoder)-1):
            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            #cls1 = x1_b[:, 0, :]   #16, 768
            #patch1 = x1_b[:, 1:, :]  #16, 196, 768
            #map1 = torch.bmm(patch1, cls1.unsqueeze(-1)).squeeze(-1)  #16, 196
            #map1 = F.softmax(map1, dim=-1)
            #cls2 = x2_b[:, 0, :]
            #patch2 = x2_b[:, 1:, :]
            #map2 = torch.bmm(patch2, cls2.unsqueeze(-1)).squeeze(-1)
            #map2 = F.softmax(map2, dim=-1)

            #cos
            cls1 = F.normalize(x1_b[:, 0, :], dim=-1)
            patch1 = F.normalize(x1_b[:, 1:, :], dim=-1)
            map1 = torch.bmm(patch1, cls1.unsqueeze(-1)).squeeze(-1)
            #map1 = (F.cosine_similarity(patch1, cls1.unsqueeze(1), dim=-1) + 1)/2
            cls2 = F.normalize(x2_b[:, 0, :], dim=-1)
            patch2 = F.normalize(x2_b[:, 1:, :], dim=-1)
            map2  = torch.bmm(patch2, cls2.unsqueeze(-1)).squeeze(-1)
            #map2 = (F.cosine_similarity(patch2, cls2.unsqueeze(1), dim=-1) + 1)/2

            # x1_c = torch.cat((cls1.unsqueeze(-2), patch1 * F.normalize(map1 - map2).unsqueeze(-1)), dim=1)
            # x2_c = torch.cat((cls2.unsqueeze(-2), patch2 * F.normalize(map2 - map1).unsqueeze(-1)), dim=1)


            x1_c = torch.cat((cls1.unsqueeze(-2), patch1 * min_max_normalize(map1 - map2).unsqueeze(-1)), dim=1)
            x2_c = torch.cat((cls2.unsqueeze(-2), patch2 * min_max_normalize(map2 - map1).unsqueeze(-1)), dim=1)

            y1 = x1_a + self.adapter_1_2_2[i - 1](x2_c, x1_b, x1_b)  #ln 이전 더하기
            y2 = x2_a + self.adapter_2_1_2[i - 1](x1_c, x2_b, x2_b)
            attn_feats['rgb'].append(y1)
            attn_feats['ir'].append(y2)
            '''
           def heatmap(feat_map):
               feat_map = feat_map[:, 1:, :]  # remove CLS token → (1, 196, 768)
               feat_map = feat_map.reshape(14, 14, 768).mean(dim=-1)
               feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
               feat_map = torch.nn.functional.interpolate(feat_map.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
               feat_map = plt.get_cmap('jet')(feat_map.cpu().numpy())  # (H, W, 4)
               feat_map = feat_map[..., :3]  # RGBA → RGB로 변환
               feat_map = torch.tensor(feat_map).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
               feat_map = feat_map.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)
               feat_map = 0.5 * original2 + 0.5 * feat_map
               plt.imshow(feat_map)
               plt.axis("off")
               plt.savefig("D:/antispoof/result/attn_map.png") '''

            y1_a = self.ln_set[i-1](y1)
            y2_a = self.ln_set[i-1](y2)

            proj1 = self.ViT_Encoder[i].mlp(y1_a) + y1
            proj2 = self.ViT_Encoder[i].mlp(y2_a) + y2

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        mlp_feats['rgb'].append(proj1[:, 0])
        mlp_feats['ir'].append(proj2[:, 0])

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)
        self.out = {
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2,
            "attn_feats": attn_feats,
            "mlp_feats": mlp_feats
        }
        return self.out

class ViT_contrast(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.num_encoders = 5
        dim = 768
        self.dim = dim
        self.ln_set = [nn.LayerNorm(768) for _ in range(self.num_encoders)]
        #self.ln1 = nn.LayerNorm(768)

        vit_b_16 = torchvision.models.vit_b_16(pretrained=True)
        for p in vit_b_16.parameters():
            p.requires_grad = True

        self.conv_proj1 = vit_b_16.conv_proj
        for p in self.conv_proj1.parameters():
            p.requires_grad = False

        self.conv_proj2 = copy.deepcopy(self.conv_proj1)

        self.class_token1 = vit_b_16.class_token
        self.class_token2 = copy.deepcopy(self.class_token1)

        vit = vit_b_16.encoder

        # self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        seq_length = 14 * 14 + 1

        self.pos_embedding1 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))
        self.pos_embedding2 = nn.Parameter(torch.empty(1, seq_length, dim).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = [vit.dropout]
        for i in range(self.num_encoders):
            if i < (self.num_encoders - 1):
                for p in vit.layers[i].parameters():
                    p.requires_grad = False
            layer_list.append(vit.layers[i])
        # add final encoder layer norm
        layer_list.append(vit.ln)

        # assign models for forward pass
        self.ViT_Encoder = nn.Sequential(*layer_list)

        self.fc_contrast = nn.Linear(2*dim, 512)

        self.fc_all = nn.Sequential(
            nn.Linear(dim + dim, 2),
        )
        self.fc_one = nn.Linear(dim, 2)

        self.adapter_1_2_2 = []
        self.adapter_2_1_2 = []

        for i in range(self.num_encoders):
            self.adapter_1_2_2.append(My_MHA_UC(dim=hidden_dim))
            self.adapter_2_1_2.append(My_MHA_UC(dim=hidden_dim))

        self.adapter_1_2_2 = nn.Sequential(*self.adapter_1_2_2)
        self.adapter_2_1_2 = nn.Sequential(*self.adapter_2_1_2)

        self.out = None

    def forward(self, x1, x2, prompts1, prompts2): # 2 d * num

        b, c, fh, fw = x1.shape
        x1 = self.conv_proj1(x1)  # b,d,gh,gw
        x1 = x1.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x2 = self.conv_proj2(x2)  # b,d,gh,gw
        x2 = x2.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x1 = torch.cat((self.class_token1.expand(b, -1, -1), x1), dim=1)  # b,gh*gw+1,d
        x2 = torch.cat((self.class_token2.expand(b, -1, -1), x2), dim=1)  # b,gh*gw+1,d

        proj1 = x1 + self.pos_embedding1
        proj2 = x2 + self.pos_embedding2

        proj1 = self.ViT_Encoder[0](proj1)
        proj2 = self.ViT_Encoder[0](proj2)

        for i in range(1, len(self.ViT_Encoder)-1): # 1, 6
            proj1 = torch.cat([proj1[:,0,:].unsqueeze(1),prompts1[i-1].unsqueeze(0).expand(b,-1,-1),proj1[:,3:,:]],dim=1)
            proj2 = torch.cat([proj2[:,0,:].unsqueeze(1),prompts2[i-1].unsqueeze(0).expand(b,-1,-1),proj2[:,3:,:]],dim=1)

            x1 = self.ViT_Encoder[i].ln_1(proj1)
            x2 = self.ViT_Encoder[i].ln_1(proj2)
            x1_c, _ = self.ViT_Encoder[i].self_attention(x1, x1, x1, need_weights=False)
            x2_c, _ = self.ViT_Encoder[i].self_attention(x2, x2, x2, need_weights=False)

            x1_a = proj1 + x1_c
            x2_a = proj2 + x2_c

            x1_b = self.ViT_Encoder[i].ln_2(x1_a)
            x2_b = self.ViT_Encoder[i].ln_2(x2_a)

            y1 = x1_a + self.adapter_1_2_2[i - 1](x2_b, x1_b, x1_b)  #ln 이전 더하기
            y2 = x2_a + self.adapter_2_1_2[i - 1](x1_b, x2_b, x2_b)

            y1_a = self.ln_set[i-1](y1)
            y2_a = self.ln_set[i-1](y2)

            proj1 = self.ViT_Encoder[i].mlp(y1_a) + y1
            proj2 = self.ViT_Encoder[i].mlp(y2_a) + y2

        proj1 = self.ViT_Encoder[-1](proj1)
        proj2 = self.ViT_Encoder[-1](proj2)

        logits1 = proj1[:, 0]  # b,d
        logits2 = proj2[:, 0]  # b,d
        logits_all = torch.cat([logits1, logits2], dim=1)
        out_contrast = self.fc_contrast(logits_all)
        out_all = self.fc_all(logits_all)
        logits1 = self.fc_one(logits1)
        logits2 = self.fc_one(logits2)

        self.out = {
            'contrast': out_contrast,
            "out_all": out_all,
            "feat_1": logits1,
            "feat_2": logits2
        }
        return self.out

class CLIP_star(nn.Module):
    def __init__(self, is_linear=None, ckpt_path='D:/antispoof/result/ca2 all/model_ckpt_49.pth'):
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        base = 'a photo of a'
        texts = ['real', 'mask', 'display', 'paper']
        self.n_ctx = 2
        self.tokens = clip.tokenize([base + t for t in texts])  # 4, 77

        self.vit = ViT_contrast()
        vit_ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.vit.load_state_dict(vit_ckpt, strict=False)

        for p in self.vit.parameters(): p.requires_grad = False

        clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model = build_model(clip_model.state_dict())
        for p in self.clip_model.parameters(): p.requires_grad = False

        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.clip_dim = self.clip_model.transformer.width
        self.vit_dim = self.vit.dim

        self.clip_layers = self.clip_model.transformer.layers
        self.vit_layers = self.vit.num_encoders + 1

        self.prompts_clip = nn.ParameterList(
            [nn.Parameter(torch.empty(self.n_ctx, self.clip_dim)) for _ in range(self.clip_layers)])
        for prompt in self.prompts_clip:nn.init.normal_(prompt,std=0.02)
        self.is_linear = is_linear
        if is_linear:
            self.proj1 = nn.Sequential(*[nn.Linear(self.clip_dim, self.vit_dim) for _ in range(self.vit_layers)])
            self.proj2 = nn.Sequential(*[nn.Linear(self.clip_dim, self.vit_dim) for _ in range(self.vit_layers)])
        else:
            self.prompts_vit1 = nn.ParameterList(
                [nn.Parameter(torch.empty(self.n_ctx, self.vit_dim)) for _ in range(self.vit_layers)])
            self.prompts_vit2 = nn.ParameterList(
                [nn.Parameter(torch.empty(self.n_ctx, self.vit_dim)) for _ in range(self.vit_layers)])
            for prompt in self.prompts_vit1: nn.init.normal_(prompt, std=0.02)
            for prompt in self.prompts_vit2: nn.init.normal_(prompt, std=0.02)

        self.class_num = {'real face': 0, 'mask attack': 1, 'display attack': 2, 'paper attack': 3}

    def forward(self, rgb, ir, label_text):
        label = torch.tensor([self.class_num[l] for l in label_text])

        logit_scale = self.logit_scale.exp()
        if self.is_linear:
            prompts_vit1 = nn.ParameterList()
            prompts_vit2 = nn.ParameterList()
            for idx in range(self.vit_layers):
                prompts_vit1.append(self.proj1[idx](self.prompts_clip[idx]))
                prompts_vit2.append(self.proj2[idx](self.prompts_clip[idx]))
        else:
            prompts_vit1 = self.prompts_vit1   #2, 768 per layer
            prompts_vit2 = self.prompts_vit2

        text_features, feats = self.clip_model.encode_text(self.tokens, self.prompts_clip)  # 4 dim=512
        image_features = self.vit(rgb, ir, prompts_vit1, prompts_vit2)  # b dim=768

        image_features = image_features['contrast']  # b d=512

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()  # b 4

        # if self.prompt_learner.training:
        return F.cross_entropy(logits, label)

if __name__ == '__main__':
    model = Swin_anti()
    input = torch.rand(size=(2,3,224,224))
    pass