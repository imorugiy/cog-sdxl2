import os
import shutil
from typing import List
from cog import BasePredictor, Path, Input
from diffusers.utils import load_image
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch

MODEL_NAME = "SG161222/Realistic_Vision_V3.0_VAE"
MODEL_CACHE = "cache"

FACE_PROMPT = "RAW photo, close up portrait of an African woman, head shot, looking at camera, aged 28, curvy, brown eyes, lips, smile, long hair"


class Predictor(BasePredictor):
    def setup(self):
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            scheduler=noise_scheduler,
            cache_dir=MODEL_CACHE,
        )

        self.pipe.to("cuda")

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
        width: int = Input(description="Width", ge=0, le=1920, default=512),
        height: int = Input(description="Height", ge=0, le=1920, default=768),
        seed: int = Input(
            description="Seed (0 = random, maximum: 2147483647)", default=0
        ),
    ) -> List[Path]:
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder="big")
        generator = torch.Generator("cuda").manual_seed(seed)

        output_paths = []

        if face_image:
            print("face image provided")
            img0 = self.load_image(face_image)
        else:
            print("face image not provided, using face prompt to generate head photo")
            output = self.pipe(
                prompt=face_prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                width=512,
                height=512,
                num_inference_steps=num_inference_steps,
            ).images

            for i, image in enumerate(output):
                output_path = f"/tmp/out-{i}.png"
                image.save(output_path)
                output_paths.append(Path(output_path))

            img0 = self.load_image("/tmp/out-0.png")

        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus-face_sd15.bin",
        )

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=img0,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images

        for i, image in enumerate(output):
            output_path = f"/tmp/out-{i+1}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
