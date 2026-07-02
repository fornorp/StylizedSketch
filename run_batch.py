import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from pipeline_curve import ADPipeline
from utils import *
import torch
import torch.nn as nn
from torch.optim import Adam

from test_curve import ClipasceneImage, ClipascenePainter, ClipasceneOptimizer

def get_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # 根据需要调整大小
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的标准化参数
    ])
    return transform(img).unsqueeze(0)  # 增加一个批次维度

def tune_vae(ref_image, model_name, device, num_epochs=75, learning_rate=1e-4):
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(
        device, dtype=torch.float32
    )
    vae.requires_grad_(False)

    image = ref_image.clone().to(device, dtype=torch.float32)
    image = image * 2 - 1
    latents = vae.encode(image)["latent_dist"].mean
    ### eval
    rec_image = vae.decode(latents, return_dict=False)[0]
    data_in = image / 2 + 0.5
    data_out = rec_image / 2 + 0.5
    diff = data_in - data_out
    diff = diff * diff
    mse = diff.mean()
    print("before training: ", mse.item())

    for param in vae.decoder.parameters():
        param.requires_grad = True
    loss_fn = nn.L1Loss()
    optimizer = Adam(vae.decoder.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        reconstructed = vae.decode(latents, return_dict=False)[0]
        loss = loss_fn(reconstructed, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item()}")

    ### eval
    rec_image = vae.decode(latents, return_dict=False)[0]
    data_in = image / 2 + 0.5
    data_out = rec_image / 2 + 0.5
    diff = data_in - data_out
    diff = diff * diff
    mse = diff.mean()
    print("after training: ", mse.item())

    return vae    


# model_name = "/root/models/stable-diffusion-v1-5"
model_name = "runwayml/stable-diffusion-v1-5"
vae = ""
lr = 0.05
iters = 1
seed = 42
width = 512
height = 512
weight = 0.25 # 0.25
batch_size = 1
mixed_precision = "bf16"
# mixed_precision = "no"
num_inference_steps = 200
guidance_scale = 1
num_images_per_prompt = 1
enable_gradient_checkpoint = False
start_layer, end_layer = 10, 16

exp_name = "exp_me"
os.makedirs(exp_name, exist_ok=True)


content_names = sorted(os.listdir("./_full_eval/content/"), key=lambda x: int(re.search(r'\d+', x).group()))
content_list = ["./_full_eval/content/" + name for name in content_names]
style_names = sorted(os.listdir("./_full_eval/style/"), key=lambda x: int(re.search(r'\d+', x).group()))
style_list = ["./_full_eval/style/" + name for name in style_names]




scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
pipe = ADPipeline.from_pretrained(
    model_name, scheduler=scheduler, safety_checker=None
)
if vae != "":
    vae = AutoencoderKL.from_pretrained(vae)
    pipe.vae = vae

pipe.classifier = pipe.unet
set_seed(seed)

cur = 0
for style_name in style_list:
    style_image = [style_name]
    style_image = torch.cat([load_image(path, size=(512, 512)) for path in style_image])
    device = torch.device("cuda:0")
    ## tuning vae
    pipe.vae = tune_vae(style_image, model_name, device)
    for content_name in content_list:
        cur += 1
        print(cur, content_name, style_name)
        img_name = content_name
        content_image = content_name
        if content_image == "":
            content_image = None
        else:
            content_image = load_image(content_image, size=(width, height))

        ## init by canny
        canny_image = load_canny(img_name, size=(width, height))
        # pidi_name = img_name.replace("content", "pidiedge")[:-4] + ".png"
        # pidi_image = load_pidi(img_name, size=(width, height))

        controller = Controller(self_layers=(start_layer, end_layer))
        

        # vec_image = VecImage(img_name, 128, device)
        # optimizer = PainterOptimizer(vec_image)
        num_paths = 128
        dataset = ClipasceneImage(img_name, num_paths, device)
        H, W = dataset.load_size()
        print(H, W)
        renderer = ClipascenePainter(num_strokes=num_paths, num_segments=1, H=H, W=W, device=device)
        renderer = renderer.to(device)
        optimizer = ClipasceneOptimizer(renderer)
        optimizer.init_optimizers()
        
        bitmap, svg, init_svg = pipe.optimize(
            lr=lr,
            batch_size=batch_size,
            iters=iters,
            width=width,
            height=height,
            weight=weight,
            controller=controller,
            style_image=style_image,
            content_image=content_image,
            mixed_precision=mixed_precision,
            num_inference_steps=num_inference_steps,
            enable_gradient_checkpoint=enable_gradient_checkpoint,
            dataset=dataset,
            renderer=renderer,
            optimizer=optimizer,
            canny_image=canny_image,
        )
        
        
        save_image(init_svg, exp_name + "/init_svg_"+str(cur)+".png")
        save_image(svg, exp_name + "/svg_"+str(cur)+".png")
        save_image(bitmap, exp_name + "/bitmap_"+str(cur)+".png")
        # show_image("style.png", title="style image")
        # show_image("content.png", title="content image")
        # show_image("output.png", title="generated")

