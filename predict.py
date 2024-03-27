import os
from typing import List
from cog import BasePredictor, Path, Input
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch

MODEL_NAME = "SG161222/Realistic_Vision_V3.0_VAE"
MODEL_CACHE = "cache"


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

    def predict(
        self,
        prompt: str = "RAW photo, close up portrait of an Jamaican woman, head shot, looking at camera, aged 28, curvy, brown eyes, lips, smile, long hair",
        negative_prompt: str = "cartoon, painting, illustration, (worst quality, low quality, normal qualiti:2)",
        num_inference_steps: int = Input(
            description="num_inference_steps", ge=0, le=100, default=30
        ),
        width: int = Input(description="Width", ge=0, le=1920, default=512),
        height: int = Input(description="Height", ge=0, le=1920, default=512),
        seed: int = Input(
            description="Seed (0 = random, maximum: 2147483647)", default=0
        ),
    ) -> List[Path]:
        if seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder="big")
        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
        ).images

        output_paths = []
        for i, image in enumerate(output):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
