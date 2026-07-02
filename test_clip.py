import collections
import clip
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def l2_layers(xs_conv_features, ys_conv_features):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None
        self.device = device

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        # featuremaps = [self.featuremaps[k] * masks_flat for k in range(12)]
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps

class CLIPConvLoss(torch.nn.Module):
    def __init__(self, device, layer=4):
        super(CLIPConvLoss, self).__init__()
        self.device = device
        self.clip_model_name = "ViT-B/32"
        self.clip_conv_loss_type = "L2"
        self.layer = layer

        self.model, self.preprocess = clip.load(
            self.clip_model_name, device, jit=False)

        self.visual_encoder = CLIPVisualEncoder(self.model, self.device)
        self.img_size = self.preprocess.transforms[1].size
        self.model.eval()

        self.normalize_transform = transforms.Compose([
            self.preprocess.transforms[0],  # Resize
            self.preprocess.transforms[1],  # CenterCrop
            self.preprocess.transforms[-1],  # Normalize
        ])

        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        # augemntations.append(transforms.RandomResizedCrop(
            # 224, scale=(0.4, 0.9), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        self.num_augs = 4

    
    def forward(self, sketch, target):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        conv_loss_dict = {}
            
        # x = self.preprocess(sketch).unsqueeze(0).to(self.device)
        # y = self.preprocess(target).unsqueeze(0).to(self.device)
        
        # x = self.normalize_transform(sketch)
        # y = self.normalize_transform(target)
        x = sketch.to(self.device)
        y = target.to(self.device)

        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
        for n in range(self.num_augs):
            augmented_pair = self.augment_trans(torch.cat([x, y]))
            sketch_augs.append(augmented_pair[0].unsqueeze(0))
            img_augs.append(augmented_pair[1].unsqueeze(0))
        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        xs_fc_features, xs_conv_features = self.visual_encoder(xs)
        ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        conv_losses = l2_layers(xs_conv_features, ys_conv_features)
        # print(conv_losses)
        result = conv_losses[self.layer]

        return result

if __name__ == "__main__":
    src = Image.open("style.png").convert("RGB")
    dst = Image.open("content.png").convert("RGB")
    device = torch.device("cuda:0")
    loss_func = CLIPConvLoss(device)
    loss = loss_func(src, dst)
    print(loss)