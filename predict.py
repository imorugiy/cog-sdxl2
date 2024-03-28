import os
import shutil
import sys

sys.path.extend(["/IP-Adapter"])

from typing import List
from PIL import Image
from cog import BasePredictor, Path, Input
from diffusers.utils import load_image
from diffusers import DDIMScheduler, StableDiffusionPipeline
from ip_adapter import IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
import torch

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE = "sdxl-cache"

image_encoder_path = "/IP-Adapter/models/image_encoder/"
ip_ckpt = "/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"
device = "cuda"

FACE_PROMPT = "RAW photo, close up portrait of an African woman, head shot, looking at camera, aged 28, curvy, brown eyes, lips, smile, long hair"


class Predictor(BasePredictor):
    def setup(self):
        self.pipe = StableDiffusionXLCustomPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            add_watermarker=False,
            cache_dir=MODEL_CACHE,
        )

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def predict(
        self,
        prompt: str = Input(
            description="image prompt",
            default="RAW photo, shot from behind of an African woman, looking at camera, aged 28, curvy, brown eyes, lips, smile, long hair, wearing black dress, within a colorful graffiti alley",
        ),
        negative_prompt: str = Input(
            description="negative image prompt",
            default="nude, NSFW, topless, cartoon, painting, illustration, (worst quality, low quality, normal qualiti:2)",
        ),
        face_prompt: str = Input(
            description="face image prompt",
            default=FACE_PROMPT,
        ),
        face_image: Path = Input(description="Face image for adapter", default=None),
        num_inference_steps: int = Input(
            description="num_inference_steps", ge=0, le=100, default=30
        ),
        width: int = Input(description="Width", ge=0, le=1920, default=1024),
        height: int = Input(description="Height", ge=0, le=1920, default=1536),
        seed: int = Input(
            description="Seed (0 = random, maximum: 2147483647)", default=0
        ),
        scale: float = Input(
            description="Scale (influence of input image on generation)",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
    ) -> List[Path]:
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder="big")

        image = Image.open(face_image)
        image.resize((512, 512))

        ip_model = IPAdapterPlusXL(
            self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=16
        )

        images = ip_model.generate(
            pil_image=image,
            num_samples=1,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            scale=scale,
        )

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
