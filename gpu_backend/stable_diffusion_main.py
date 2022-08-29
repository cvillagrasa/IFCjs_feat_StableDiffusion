import numpy as np
import torch
import torchcsprng as csprng
from torch.cuda.amp import autocast
import PIL
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from tqdm.auto import tqdm
import inspect
from typing import List, Optional, Union
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from utils import SingletonMeta


generator = csprng.create_random_device_generator('/dev/urandom')
device = 'cuda'


class StableDiffusionInpaintingPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: torch.FloatTensor,
        mask_image: torch.FloatTensor,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
    ):

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        init_latents = self.vae.encode(init_image.to(self.device)).sample()
        init_latents = 0.18215 * init_latents
        init_latents_orig = init_latents
        mask = mask_image.to(self.device)
        init_latents = torch.cat([init_latents] * batch_size)
        init_timestep = int(num_inference_steps) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]
            my_noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
            init_latents_proper = self.scheduler.add_noise(init_latents_orig, my_noise, t)
            latents = ( init_latents_proper * mask ) + ( latents * (1-mask) )

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        has_nsfw_concept = 0

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}


class PipeImg(metaclass=SingletonMeta):
    def __init__(self):
        self.model = self.setup_inpainting_model()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def setup_inpainting_model(self):
        pipeimg = StableDiffusionInpaintingPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True
        ).to(device)
        pipeimg.safety_checker = self.dummy_nsfw_safety
        return pipeimg

    def dummy_nsfw_safety(images, **kwargs):
        return images, False


def preprocess_img(image):
    image = image.resize((512, 512))
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    mask = mask.convert("L")
    mask = mask.resize((64, 64), resample=PIL.Image.LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask = torch.from_numpy(mask)
    return mask


def infer_from_gradio(prompt, img, samples_num, steps_num, scale, option):
    if option == "Replace selection":
        mask = 1 - preprocess_mask(img["mask"])
    else:
        mask = preprocess_mask(img["mask"])

    img = preprocess_img(img["image"])
    pipeimg = PipeImg()

    with autocast():
        images = pipeimg(
            [prompt] * samples_num, init_image=img, mask_image=mask, num_inference_steps=steps_num,
            guidance_scale=scale, generator=generator
        )["sample"]
    return images


def infer(prompt, img, steps=250, scale=50, thres=235):
    pixel_to_b_or_w = lambda x: 0 if x > thres else 255
    img_mask = img.convert('L').point(pixel_to_b_or_w, mode='1').convert('1')
    mask = preprocess_mask(img_mask)
    img = img.convert('RGB')
    img = preprocess_img(img)
    pipeimg = PipeImg()

    with autocast():
        images = pipeimg(
            prompt, init_image=img, mask_image=mask, num_inference_steps=steps, guidance_scale=scale,
            generator=generator
        )["sample"]
    return images[0]


if __name__ == "__main__":
    img = Image.open('~/test.png')
    prompt = 'Photograph of my office in the Sahara desert, surrounded by some camels. Sunny day. National Geographic.'
    img_result = infer(prompt, img)
    img_result.save('test_generated.png')
