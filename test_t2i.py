import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
from pipeline_sd import ADPipeline
from diffusers import DDIMScheduler, AutoencoderKL
import torch
from utils import *
from accelerate.utils import set_seed


# model_name = "/root/models/stable-diffusion-v1-5"
model_name = "runwayml/stable-diffusion-v1-5"
vae = ""
lr = 0.015
iters = 3
seed = 42
mixed_precision = "bf16"
num_inference_steps = 50
guidance_scale = 7.5
num_images_per_prompt = 1
enable_gradient_checkpoint = False
start_layer, end_layer = 10, 16

prompt = "A photo of a car behind a wooden house"
style_image = ["./tests/contour_width3.png"]

scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
pipe = ADPipeline.from_pretrained(
    model_name, scheduler=scheduler, safety_checker=None
)
if vae != "":
    vae = AutoencoderKL.from_pretrained(vae)
    pipe.vae = vae

pipe.classifier = pipe.unet
set_seed(seed)

style_image = torch.cat([load_image(path, size=(512, 512)) for path in style_image])
controller = Controller(self_layers=(start_layer, end_layer))

result = pipe.sample(
    controller=controller,
    iters=iters,
    lr=lr,
    adain=True,
    height=512,
    width=512,
    mixed_precision="bf16",
    style_image=style_image,
    prompt=prompt,
    negative_prompt="",
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    num_images_per_prompt=num_images_per_prompt,
    enable_gradient_checkpoint=enable_gradient_checkpoint
)

save_image(style_image, "style.png")
save_image(result, "output.png")
# show_image("style.png", title="style image")
# show_image("output.png", title=prompt)
