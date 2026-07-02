import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from pipeline_sd import ADPipeline
from utils import *

# model_name = "/root/models/stable-diffusion-v1-5"
model_name = "runwayml/stable-diffusion-v1-5"
vae = ""
# lr = 0.05
lr = 0.05
iters = 1
seed = 42
width = 512
height = 512
weight = 0.25
batch_size = 1
mixed_precision = "bf16"
num_inference_steps = 200
guidance_scale = 1
num_images_per_prompt = 1
enable_gradient_checkpoint = False
# start_layer, end_layer = 10, 16
start_layer, end_layer = 10, 16

style_image = ["./tests/opensketch_Prof5.jpg"]
content_image = "./data/content/12.jpg"

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
if content_image == "":
    content_image = None
else:
    content_image = load_image(content_image, size=(width, height))
controller = Controller(self_layers=(start_layer, end_layer))
result = pipe.optimize(
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
)

save_image(style_image, "style.png")
save_image(content_image, "content.png")
save_image(result, "output.png")
# show_image("style.png", title="style image")
# show_image("content.png", title="content image")
# show_image("output.png", title="generated")
