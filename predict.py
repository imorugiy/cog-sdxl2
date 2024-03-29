import os
import shutil
from typing import List
from cog import BasePredictor, Path, Input
from insightface.app import FaceAnalysis
from diffusers.utils import load_image
from diffusers import DDIMScheduler, StableDiffusionPipeline, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
import torch
import cv2

MODEL_NAME = "SG161222/Realistic_Vision_V4.0_noVAE"
MODEL_CACHE = "cache"

VAE_NAME = "stabilityai/sd-vae-ft-mse"
VAE_CACHE = "vae-cache"

IP_CACHE = "ip-cache"
ip_ckpt = "ip-cache/ip-adapter-faceid_sd15.bin"

FACE_PROMPT = "RAW photo, a woman, african, head shot, looking at camera, 28 years old, curvy, brown eyes, lips, smile, long hair"


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

        vae = AutoencoderKL.from_pretrained(VAE_NAME, cache_dir=VAE_CACHE).to(
            dtype=torch.float16
        )

        face_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            cache_dir=MODEL_CACHE,
        )
        self.face_pipe = face_pipe.to("cuda")

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            cache_dir=MODEL_CACHE,
        )
        self.pipe = pipe.to("cuda")
        self.ip_model = IPAdapterFaceID(pipe, ip_ckpt, "cuda")

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def predict(
        self,
        prompt: str = Input(
            description="image prompt",
            default="A photo of a woman, full body shot, wearing black dress, in a colorful graffiti alley, 28 years old, curvy, brown eyes, lips, smile, long hair",
        ),
        negative_prompt: str = Input(
            description="negative image prompt",
            default="bokeh, monochrome, lowres, bad anatomy, worst quality, low quality, blurry",
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
            img0 = face_image
        else:
            print("face image not provided, using face prompt to generate head photo")
            output = self.face_pipe(
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

            img0 = output_paths[0]

        print("Extracting face embeddings: ", img0)
        self.app.prepare(ctx_id=0, det_size=(512, 512))
        image = cv2.imread(str(img0))
        faces = self.app.get(image)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=1,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i+1}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
