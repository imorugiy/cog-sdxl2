import os
import shutil
import sys

sys.path.extend(["/IP-Adapter"])

import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from typing import List
from PIL import Image
from cog import BasePredictor, Path, Input
from diffusers.utils import load_image
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from ip_adapter import IPAdapterPlusXL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL, IPAdapterFaceIDPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
import torch

MODEL_NAME = "SG161222/RealVisXL_V3.0"
MODEL_CACHE = "sdxl-cache"

IP_CACHE = "ip-cache"

image_encoder_path = "/IP-Adapter/models/image_encoder/"
ip_ckpt_face_id = "ip-cache/ip-adapter-faceid_sdxl.bin"
ip_ckpt = "/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin"
device = "cuda"

FACE_PROMPT = "RAW photo, close up portrait of an African woman, head shot, looking at camera, aged 28, curvy, brown eyes, lips, smile, long hair"


class Predictor(BasePredictor):
    def setup(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            add_watermarker=False,
            cache_dir=MODEL_CACHE,
        )

        self.pipe = pipe.to(device)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def predict(
        self,
        prompt: str = Input(
            description="image prompt",
            default="A photo, an african woman, full body shot, wearing swimming suit, on a sunny beach, looking at camera, aged 28, curvy, brown eyes, lips, smile, long hair",
        ),
        negative_prompt: str = Input(
            description="negative image prompt",
            default="nude, NSFW, topless, (worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
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
        generator = torch.Generator("cuda").manual_seed(seed)

        output_paths = []
        img0 = face_image
        if not face_image:
            print("face image not provided, generating one")
            output = self.pipe(
                prompt=face_prompt,
                negative_prompt=negative_prompt,
                width=1024,
                height=1024,
                generator=generator,
                num_inference_steps=num_inference_steps,
            ).images

            for i, image in enumerate(output):
                output_path = f"/tmp/out-{i}.png"
                image.save(output_path)
                output_paths.append(Path(output_path))

            img0 = output_paths[0]

        self.app.prepare(ctx_id=0, det_size=(512, 512))
        image = cv2.imread(str(img0))
        faces = self.app.get(image)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        ip_model = IPAdapterFaceIDXL(self.pipe, ip_ckpt_face_id, device)
        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=1,
            seed=seed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            scale=1.0,
        )

        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i+1}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
