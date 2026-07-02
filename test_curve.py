import pydiffvg
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import models, transforms

from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

import os
import cv2
from PIL import Image
import numpy as np
import random

from test_clip import CLIPConvLoss

class ClipasceneImage(Dataset):
    def __init__(self, data_path, num_paths, device):
        self.data_path = data_path
        self.device = device
        self.num_paths = num_paths
        self.shapes = []
        self.points_init = []
        self.masks = []
        self.data_transforms = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()]) 
        self.H = 0
        self.W = 0
        self.get_attention(num_paths, device)
        
    def get_attention(self, num_paths, device):
        path = self.data_path 
        target = Image.open(path).convert("RGB")
        target_ = self.data_transforms(target).unsqueeze(0)
        self.H = target_.shape[2]
        self.W = target_.shape[3]
        attn_map = np.ones((self.H, self.W))
        xdog = XDoG_()
        im_xdog = xdog(target_[0].permute(1,2,0).cpu().numpy(), k=10)
        attn_map = (1 - im_xdog) * attn_map
        # cv2.imwrite('xdog.png', attn_map*255)
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0])
        k = num_paths
        inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        inds = np.array(np.unravel_index(inds, attn_map.shape)).T    
        inds_normalised = np.zeros(inds.shape)
        inds_normalised[:, 0] = inds[:, 1] / self.W
        inds_normalised[:, 1] = inds[:, 0] / self.H
        inds_normalised = inds_normalised.tolist()
        self.init_curve(num_paths, inds_normalised, device)
        print('get attn done')

    def init_curve(self, num_paths, inds_normalised, device):
        shapes = []
        points_init = []
        for i in range(num_paths):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = self.get_path(points_init, i, inds_normalised, device)
            shapes.append(path)
        self.shapes.append(shapes)
        self.points_init.append(points_init)

    def get_path(self, points_init, counter, inds_normalised, device):
        points = []
        p0 = inds_normalised[counter]
        points.append(p0)
        radius = 0.05
        for k in range(3):
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            points.append(p1)
            p0 = p1
        points = torch.tensor(points).to(device)
        points[:, 0] *= self.W
        points[:, 1] *= self.H
        points_init.append(points)
        num_control_points = torch.zeros(1, dtype=torch.int32) + 2
        path = pydiffvg.Path(num_control_points = num_control_points,
                                points = points,
                                stroke_width = torch.tensor(1.5).to(device),
                                is_closed = False)
        return path

    def softmax(self, x, tau=0.3):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 
    
    def __getitem__(self, idx):
        path = os.path.join(self.data_path)
        target = Image.open(path).convert("RGB")
        target_ = self.data_transforms(target).unsqueeze(0)
        return target_, self.points_init[idx]

    def load_size(self):
        return self.H, self.W

    def __len__(self):
        return 1

class ClipascenePainter(torch.nn.Module):
    def __init__(self,
                num_strokes=64,
                num_segments=1,
                H = 512,
                W = 512,
                device=None):
        super(ClipascenePainter, self).__init__()

        self.num_paths = num_strokes
        self.num_segments = 1
        self.width = 1.5
        self.control_points_per_seg = 4
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)
        self.device = device
        self.canvas_width = W
        self.canvas_height = H
        self.shapes = []
        self.shape_groups = []
        self.mlp = MLP(num_strokes=self.num_paths, num_cp=self.control_points_per_seg).to(device)
        # self.tnet = TNet(self.num_paths, 128, 2)
        # self.tnet.apply(init_weights)

        self.re_init = False
        self.pred = None
        self.train_color = False
    
    def get_image(self, points_init):
        img, points_opt = self.mlp_pass(points_init)
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img, points_opt, self.shapes, self.shape_groups

    def re_initialize(self):
        self.re_init = True
        self.new_init = self.pred.clone().detach()
        self.mlp.apply(init_weights)
    
    def mlp_pass(self, points_init, eps=1e-4):
        """
        update self.shapes etc through mlp pass instead of directly (should be updated with the optimizer as well).
        """
        if self.re_init == False:
            points_vars = points_init
            points_vars = torch.stack(points_vars).unsqueeze(0).to(self.device)
        else:
            points_vars = self.new_init.clone()
        points_vars[:,:,:,0] = points_vars[:,:,:,0] / self.canvas_width
        points_vars[:,:,:,1] = points_vars[:,:,:,1] / self.canvas_height
        points_vars = 2 * points_vars - 1
        points, widths, colors = self.mlp(points_vars)
        # points, widths, colors = self.tnet(points_vars)
        # print(points.shape, widths.shape)
                    
        # normalize back to canvas size [0, 224] and reshape
        all_points = 0.5 * (points + 1.0)
        all_points = all_points.reshape((-1, self.num_paths, self.control_points_per_seg, 2))
        all_points[:,:,:,0] *= self.canvas_width
        all_points[:,:,:,1] *= self.canvas_height

        self.pred = all_points.clone().detach()

        # define new primitives to render
        shapes = []
        shape_groups = []
        default_color = torch.tensor([0,0,0,1]).cuda()
        
        for p in range(self.num_paths):
            # width = torch.tensor(self.width)
            # width = widths[0,p] * self.width * 2.0
            # width = widths[0,p] * self.width # for thin sketches
            # if self.train_color == False:
            #     width = torch.tensor(self.width).cuda() * 0.5
            # else:
            ## fxn: change the width coefficient
            width = widths[0,p] * self.width * 2.0
            # color = default_color
            color = default_color * colors[0,p]
            path = pydiffvg.Path(
                num_control_points=self.num_control_points, points=all_points[:,p].reshape((-1,2)),
                stroke_width=width, is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)
        
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, shapes, shape_groups)
        img = _render(self.canvas_width, # width
                    self.canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        self.shapes = shapes.copy()
        self.shape_groups = shape_groups.copy()
        return img, all_points
        
    def parameters(self):
        self.points_vars = self.mlp.parameters()
        # self.points_vars = self.tnet.parameters()
        return self.points_vars
    
    def get_mlp(self):
        return self.mlp
        # return self.tnet
    
    def get_points_params(self):
        return dict(list(self.mlp.named_parameters()))
        # return dict(list(self.tnet.named_parameters()))
        # return self.points_vars
    
    def save_svg(self, output_dir, name, idx, shapes, shape_groups):
        pydiffvg.save_svg('{}/{}-{}.svg'.format(output_dir, name, str(idx)), self.canvas_width, self.canvas_height, shapes, shape_groups)
        
class MLP(nn.Module):
    def __init__(self, num_strokes, num_cp):
        super().__init__()
        outdim = 1000
        # outdim = 2048
        self.layers_points = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_strokes * num_cp * 2, outdim),
            nn.SELU(inplace=True),
            nn.Linear(outdim, outdim),
            nn.SELU(inplace=True),
            # nn.Tanh()
        )
        self.offset_head = nn.Linear(outdim, num_strokes * num_cp * 2)
        self.width_head = nn.Linear(outdim, num_strokes)
        self.color_head = nn.Linear(outdim, num_strokes)
        self.act = nn.Sigmoid()

    def forward(self, x, widths=None):
        '''Forward pass'''
        latent = self.layers_points(x)
        deltas = self.offset_head(latent)
        widths = self.act(self.width_head(latent))
        colors = self.act(self.color_head(latent))
        # return deltas
        return x.flatten() + 0.1 * deltas, widths, colors

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
def positional_encoding(x, num_freqs):
    # x: N * 2
    x = x.unsqueeze(-1)
    dim = x.shape[-2]
    device = torch.device("cuda:0")
    scales = 2**torch.arange(num_freqs, dtype=torch.float32).to(device)
    positions = x * scales
    embeddings = torch.cat([torch.sin(positions), torch.cos(positions)], dim=-1)
    return embeddings

class PtEmbed(nn.Module):
    def __init__(self, embed_dim=64, num_freqs=3):
        super().__init__()
        self.num_freqs = num_freqs
        input_dim = 8 * (num_freqs * 2 + 1)
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, pt):
        num_paths = pt.shape[1]
        pt_in = pt.reshape((-1, 2))
        pe = positional_encoding(pt_in, self.num_freqs)
        pos_in = pt.reshape((-1, 8))
        pe_in = pe.reshape(num_paths, -1)
        embed_in = torch.cat([pos_in, pe_in], dim=-1)
        embed_out = self.proj(embed_in)
        return embed_out


class TNet(nn.Module):
    def __init__(self, num_paths=128, embed_dim=64, head=2):
        super().__init__()
        self.num_paths = num_paths
        self.embed_proj = PtEmbed(embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=head, dim_feedforward=embed_dim*4, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.pred_head = nn.Linear(embed_dim, 8)
        self.pred_width = nn.Linear(embed_dim, 1)
        self.pred_color = nn.Linear(embed_dim, 1)
        
    def forward(self, pt):
        # print(embed.dtype, pt.dtype)
        embed = self.embed_proj(pt)
        feature = self.encoder(embed)
        fs = feature.squeeze(1)
        offset = self.pred_head(fs)
        offset = offset.reshape((1, -1, 4, 2))
        result = pt + 0.1 * offset
        widths = F.sigmoid(self.pred_width(fs))
        widths = widths.view(1, -1)
        # color = 1.0 - F.relu(1.0 - F.relu(self.pred_color(fs)))
        color = F.sigmoid(self.pred_color(fs))
        color = color.view(1, -1)
        return result, widths, color


class ClipasceneOptimizer:
    def __init__(self, renderer):
        self.renderer = renderer
        self.points_lr = 1e-4 # for mlp
        # self.points_lr = 1e-2 # for tnet
        self.points_optim = None

    def turn_off_points_optim(self):
        self.optimize_points = False

    def init_optimizers(self):
        points_params = self.renderer.parameters()        
        self.points_optim = torch.optim.Adam(points_params, lr=self.points_lr)

    def update_lr(self, counter):
        new_lr = utils.get_epoch_lr(counter, self.args)
        for param_group in self.points_optim.param_groups:
            param_group["lr"] = new_lr
    
    def zero_grad_(self):
        self.points_optim.zero_grad()
    
    def step_(self):
        self.points_optim.step()
    
    def get_lr(self, optim="points"):
        return self.points_optim.param_groups[0]['lr']

    def get_points_optim(self):
        return self.points_optim
    
class ClipassoImage(torch.nn.Module):
    def __init__(self, img_path, num_paths, device):
        self.img_path = img_path
        self.num_paths = num_paths
        self.data_transforms = transforms.Compose([transforms.ToTensor()]) 
        self.device = device
        self.shapes = []
        self.shape_groups = []
        # self.zcolor = torch.zeros(num_paths, 1).to(device)
        # self.points_init = []
        self.H = 512
        self.W = 512
        self.curve_width = 1.5
        self.get_attention(img_path, num_paths)
        
    def get_attention(self, img_path, num_paths):
        target = Image.open(img_path).convert("RGB")
        target_ = self.data_transforms(target).unsqueeze(0)
        if target_.shape[2] != self.H or target_.shape[3] != self.W:
            target_ = F.interpolate(target_, (self.H, self.W), mode='bilinear')
        xdog = XDoG_()
        im_xdog = xdog(target_[0].permute(1,2,0).cpu().numpy(), k=10)
        attn_map = 1 - im_xdog
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0])
        k = num_paths
        inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        inds = np.array(np.unravel_index(inds, attn_map.shape)).T    
        inds_normalised = np.zeros(inds.shape)
        inds_normalised[:, 0] = inds[:, 1] / self.W
        inds_normalised[:, 1] = inds[:, 0] / self.H
        inds_normalised = inds_normalised.tolist()
        self.init_curve(num_paths, inds_normalised)

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 

    def init_curve(self, num_paths, inds_normalised):
        self.shapes = []
        # self.points_init = []
        for i in range(num_paths):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = self.get_path(i, inds_normalised)
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                             fill_color = None,
                                             stroke_color = stroke_color)
            self.shape_groups.append(path_group)

    def get_path(self, counter, inds_normalised):
        points = []
        p0 = inds_normalised[counter]
        points.append(p0)
        radius = 0.05
        for k in range(3):
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            points.append(p1)
            p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.W
        points[:, 1] *= self.H
        # self.points_init.append(points)
        num_control_points = torch.zeros(1, dtype=torch.int32) + 2
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(1.5).to(self.device),
                             is_closed=False)
        return path

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - opacity)
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img

    def render_warp(self):
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.W, self.H, self.shapes, self.shape_groups)
        img = _render(self.W, # width
                    self.H, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        return img

    def parameters(self):
        self.points_vars = []
        self.width_vars = []
        for i, path in enumerate(self.shapes):
            path.points.requires_grad = True
            self.points_vars.append(path.points)
            path.stroke_width.requires_grad = True
            self.width_vars.append(path.stroke_width)
        return self.points_vars, self.width_vars

    def save_svg(self, output_dir, name, idx):
        pydiffvg.save_svg('{}/{}-{}.svg'.format(output_dir, name, str(idx)), 
                          self.W, 
                          self.H, 
                          self.shapes, 
                          self.shape_groups)

class ClipassoOptimizer:
    def __init__(self, renderer):
        self.renderer = renderer
        self.points_lr = 1.0
        points_params, width_params = self.renderer.parameters()        
        self.points_optim = torch.optim.Adam([{'params':points_params,'lr':1.0},
                                              {'params':width_params,'lr':0.01}], lr=0.01)
    
    def zero_grad_(self):
        self.points_optim.zero_grad()
    
    def step_(self):
        self.points_optim.step()
    
    def get_lr(self, optim="points"):
        return self.points_optim.param_groups[0]['lr']

    def get_points_optim(self):
        return self.points_optim

class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True
        
    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)        

if __name__ == "__main__":
    '''
    x = torch.rand(1, 128, 4, 2)
    net = TNet(128, 128, 1)
    y, w, c = net(x)
    print(y.shape, w.shape, c.shape)
    print(x)
    print(y)
    print(w)
    print(c)
    '''
    
    '''
    img_name = "13.png"
    device = torch.device("cuda:0")
    vec_image = ClipassoImage(img_name, 64, device)
    # img = vec_image.get_image()
    # img_np = img[0].cpu().numpy().transpose((1, 2, 0))
    # cv2.imwrite("init.png", img_np*255)
    optimizer = ClipassoOptimizer(vec_image)
    optimizer.zero_grad_()
    img = vec_image.get_image()
    loss = img.mean()
    loss.backward()
    print(loss.item())
    optimizer.step_()
    '''

    '''
    img_name = "data/content/1.jpg"
    num_paths = 64
    device = torch.device("cuda:0")
    dataset = ClipasceneImage(img_name, num_paths, device)
    H, W = dataset.load_size()
    renderer = ClipascenePainter(num_strokes=num_paths,
                       num_segments=1,
                       H = H,
                       W = W,
                       device=device)
    renderer = renderer.to(device)
    optimizer = ClipasceneOptimizer(renderer)
    with torch.no_grad():
        inputs, points_init = dataset.__getitem__(0)
        init_sketches, points_opt, shapes, shape_groups = renderer.get_image(points_init)
        init_sketches = init_sketches.to(device)
        renderer.save_svg(f"tests", f"init", 0, shapes, shape_groups)
        points_save = points_opt.detach().cpu()
        torch.save({'points': points_save}, 'init_pt.pth')
    '''

    
    for k in range(1):
        img_name = "./joint/0712/bitmap_" + str(k+1) + ".png"
        device = torch.device("cuda:0")
        # vec_image = VecImage(img_name, 128, device)
        # optimizer = PainterOptimizer(vec_image)
        num_paths = 128
        dataset = ClipasceneImage(img_name, num_paths, device)
        H, W = dataset.load_size()
        renderer = ClipascenePainter(num_strokes=num_paths, num_segments=1, H=H, W=W, device=device)
        renderer = renderer.to(device)
        optimizer = ClipasceneOptimizer(renderer)
        optimizer.init_optimizers()    
    
        with torch.no_grad():
            inputs, points_init = dataset.__getitem__(0)
            init, points_opt, shapes, shape_groups = renderer.get_image(points_init)
            renderer.save_svg(f"0712_vec_tnet", f"init", k+1, shapes, shape_groups)
        
        clip_loss = CLIPConvLoss(device)
        clip_loss_2 = CLIPConvLoss(device, 2)
        loss_fn = torch.nn.MSELoss()
        
        for i in range(1000):
            inputs, points_init = dataset.__getitem__(0)
            cur_img, points_opt, shapes, shape_groups = renderer.get_image(points_init)
            cur_img = cur_img.to(device)
            if i < 300:
                perceptual_loss = clip_loss(cur_img, inputs.to(device))
            elif i < 600:
                perceptual_loss = clip_loss_2(cur_img, inputs.to(device))
            else:
                perceptual_loss = loss_fn(cur_img, inputs.to(device))
            optimizer.zero_grad_()
            perceptual_loss.backward()
            optimizer.step_()
            if i % 100 == 0:
                print(i, perceptual_loss.item())
            # if i == 500:
            #     renderer.re_initialize()
    
        with torch.no_grad():
            inputs, points_init = dataset.__getitem__(0)
            result, points_opt, shapes, shape_groups = renderer.get_image(points_init)
            renderer.save_svg(f"0712_vec_tnet", f"final", k+1, shapes, shape_groups)
    




