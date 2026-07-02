import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

loss_fn = torch.nn.L1Loss()
loss_fxn = torch.nn.MSELoss()


def ad_loss(
    q_list, ks_list, vs_list, self_out_list, scale=1, source_mask=None, target_mask=None
):
    loss = 0
    attn_mask = None
    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        if source_mask is not None and target_mask is not None:
            w = h = int(np.sqrt(q.shape[2]))
            mask_1 = torch.flatten(F.interpolate(source_mask, size=(h, w)))
            mask_2 = torch.flatten(F.interpolate(target_mask, size=(h, w)))
            attn_mask = mask_1.unsqueeze(0) == mask_2.unsqueeze(1)
            attn_mask=attn_mask.to(q.device)

        target_out = F.scaled_dot_product_attention(
            q * scale,
            torch.cat(torch.chunk(ks, ks.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            torch.cat(torch.chunk(vs, vs.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            attn_mask=attn_mask
        )
        loss += loss_fn(self_out, target_out)
        # print(self_out.shape, q.shape, ks.shape, vs.shape)
        ## original loss with detach
        # loss += loss_fn(self_out, target_out.detach())
        # break
        # loss += loss_fn(target_out, self_out.detach())
    # exit()
    return loss

def ad_loss_list(q_list, ks_list, vs_list, self_out_list, scale=1, source_mask=None, target_mask=None):
    losses = []
    attn_mask = None
    for q, ks, vs, self_out in zip(q_list, ks_list, vs_list, self_out_list):
        if source_mask is not None and target_mask is not None:
            w = h = int(np.sqrt(q.shape[2]))
            mask_1 = torch.flatten(F.interpolate(source_mask, size=(h, w)))
            mask_2 = torch.flatten(F.interpolate(target_mask, size=(h, w)))
            attn_mask = mask_1.unsqueeze(0) == mask_2.unsqueeze(1)
            attn_mask=attn_mask.to(q.device)

        target_out = F.scaled_dot_product_attention(
            q * scale,
            torch.cat(torch.chunk(ks, ks.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            torch.cat(torch.chunk(vs, vs.shape[0]), 2).repeat(q.shape[0], 1, 1, 1),
            attn_mask=attn_mask
        )
        losses.append(loss_fn(self_out, target_out))
    return losses

def dino_loss(dino, src, dst, last_i=1):
    transform = transforms.Compose([
        transforms.Resize(560, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
        
    src_ = transform(src)
    dst_ = transform(dst)
    src_feature = dino.get_intermediate_layers(src_, 8)[0] # layer 4
    dst_feature = dino.get_intermediate_layers(dst_, 8)[0]
    # print(src_feature.shape, src_feature.mean(), src_feature.abs().mean())
    # print(dst_feature.shape, dst_feature.mean(), dst_feature.abs().mean())
    # exit()
    # loss = loss_fxn(src_feature, dst_feature)
    loss = torch.square(src_feature - dst_feature).mean()
    return loss

def q_loss(q_list, qc_list):
    loss = 0
    for q, qc in zip(q_list, qc_list):
        loss += loss_fn(q, qc.detach())
    return loss

def qmin_loss(q_list, qv_list, qe_list):
    loss = 0
    for q, qv, qe in zip(q_list, qv_list, qe_list):
        diff_1 = torch.abs(q - qv.detach()).mean(dim=3).mean(dim=1)
        diff_2 = torch.abs(q - qe.detach()).mean(dim=3).mean(dim=1)
        diff = torch.cat([diff_1, diff_2], dim=0)
        # print(diff.shape)
        diff_min, _ = diff.min(dim=0, keepdim=True)
        loss += diff_min.mean()
    return loss

def weighted_q_loss(q_list, qv_list, v_out_list, qe_list, e_out_list, ks_list, vs_list):
    loss = 0
    layers = len(q_list)
    for i in range(layers):
        qv, vout = qv_list[i], v_out_list[i]
        qe, eout = qe_list[i], e_out_list[i]
        ks, vs = ks_list[i], vs_list[i]

        with torch.no_grad():
            target_v_s = F.scaled_dot_product_attention(
                qv,
                torch.cat(torch.chunk(ks, ks.shape[0]), 2).repeat(qv.shape[0], 1, 1, 1),
                torch.cat(torch.chunk(vs, vs.shape[0]), 2).repeat(qv.shape[0], 1, 1, 1),
                attn_mask=None
            )
            diff_v_s = torch.abs(vout - target_v_s).mean(dim=3).mean(dim=1)
            target_e_s = F.scaled_dot_product_attention(
                qe,
                torch.cat(torch.chunk(ks, ks.shape[0]), 2).repeat(qe.shape[0], 1, 1, 1),
                torch.cat(torch.chunk(vs, vs.shape[0]), 2).repeat(qe.shape[0], 1, 1, 1),
                attn_mask=None
            )
            diff_e_s = torch.abs(eout - target_e_s).mean(dim=3).mean(dim=1)
            weight = torch.cat([diff_v_s, diff_e_s], dim=0)
            weight = torch.softmax(weight, dim=0)
        
        q = q_list[i]
        diff_1 = torch.abs(q - qv.detach()).mean(dim=3).mean(dim=1)
        diff_2 = torch.abs(q - qe.detach()).mean(dim=3).mean(dim=1)
        diff = torch.cat([diff_1, diff_2], dim=0)
        diff_weighted = (1.0 - weight) * diff
        m_dist = diff_weighted.mean()
        loss += m_dist
        # print(diff.shape, weight.shape, m_dist.item())
    # exit()
    return loss

# weight = 200
def qk_loss(q_list, k_list, qc_list, kc_list):
    loss = 0
    for q, k, qc, kc in zip(q_list, k_list, qc_list, kc_list):
        scale_factor = 1 / math.sqrt(q.size(-1))
        self_map = torch.softmax(q @ k.transpose(-2, -1) * scale_factor, dim=-1)
        target_map = torch.softmax(qc @ kc.transpose(-2, -1) * scale_factor, dim=-1)
        loss += loss_fn(self_map, target_map.detach())
    return loss

# weight = 1
def qkv_loss(q_list, k_list, vc_list, c_out_list):
    loss = 0
    for q, k, vc, target_out in zip(q_list, k_list, vc_list, c_out_list):
        self_out = F.scaled_dot_product_attention(q, k, vc)
        loss += loss_fn(self_out, target_out.detach())
    return loss
