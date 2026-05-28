from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.diffusion import DEFAULT_DIFFUSION_STEPS, ddim_denoise_step, inference_timesteps
from vlm_diffadapter.modeling import VlmDiffAdapter
from vlm_diffadapter.training import load_checkpoint


@dataclass(frozen=True)
class MultimodalGenerationResult:
    text: str
    image: Image.Image
    image_prompt: str


def load_model(config: str | Path, checkpoint: str | Path | None = None) -> VlmDiffAdapter:
    model = VlmDiffAdapter(load_model_config(config))
    if checkpoint is not None:
        load_checkpoint(checkpoint, model=model)
    return model


def generate_caption(
    model: VlmDiffAdapter,
    image: Image.Image,
    prompt: str | None = None,
    generation_config: dict[str, object] | None = None,
) -> str:
    config = generation_config or {}
    if _is_causal_text_tower(model) and (
        model.visual_text_adapter is not None or model.xfusion_adapter is not None
    ):
        return _generate_causal_visual_prefix_caption(model, image, prompt=prompt, generation_config=config)
    if model.visual_text_adapter is not None:
        return _generate_visual_prefix_caption(model, image, prompt=prompt, generation_config=config)
    del model, generation_config
    prefix = prompt or "Describe the image"
    return f"{prefix}: image size {image.width}x{image.height}."


def generate_image(
    model: VlmDiffAdapter,
    prompt: str,
    generation_config: dict[str, object] | None = None,
    seed: int | None = None,
    size: tuple[int, int] = (64, 64),
) -> Image.Image:
    config = generation_config or {}
    num_inference_steps = int(config.get("num_inference_steps", 16))
    text_length = int(config.get("text_length", 32))
    num_train_timesteps = int(config.get("num_train_timesteps", DEFAULT_DIFFUSION_STEPS))
    guidance_scale = float(config.get("guidance_scale", 1.0))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    latents = torch.randn(
        1,
        model.config.image_channels,
        model.config.image_size,
        model.config.image_size,
        device=device,
        generator=generator,
    )
    tokens = _tokenize_prompt(prompt, text_length=text_length, vocab_size=model.vocab_size).unsqueeze(0).to(device)
    unconditional_tokens = _tokenize_prompt("", text_length=text_length, vocab_size=model.vocab_size).unsqueeze(0).to(device)
    timesteps = inference_timesteps(
        num_inference_steps=num_inference_steps,
        num_train_timesteps=num_train_timesteps,
        device=device,
        schedule=model.config.diffusion_schedule,
    )
    with torch.no_grad():
        for index, timestep in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else torch.zeros((), dtype=torch.long, device=device)
            timestep_batch = timestep.expand(latents.shape[0])
            previous_batch = previous.expand(latents.shape[0])
            noise_pred = _predict_guided_noise(
                model=model,
                latents=latents,
                tokens=tokens,
                unconditional_tokens=unconditional_tokens,
                timestep_batch=timestep_batch,
                guidance_scale=guidance_scale,
            )
            latents = ddim_denoise_step(
                latents=latents,
                predicted_noise=noise_pred,
                timestep=timestep_batch,
                previous_timestep=previous_batch,
                num_train_timesteps=num_train_timesteps,
                schedule=model.config.diffusion_schedule,
            )
        decoded = model.vae.decode(latents).detach().cpu()[0]
    if was_training:
        model.train()
    return _tensor_to_image(decoded, size=size)


def generate_multimodal(
    model: VlmDiffAdapter,
    image: Image.Image,
    prompt: str,
    generation_config: dict[str, object] | None = None,
    seed: int | None = None,
    size: tuple[int, int] = (64, 64),
    image_prompt_mode: str = "prompt-answer",
) -> MultimodalGenerationResult:
    config = generation_config or {}
    caption_config = config.get("caption", config)
    image_config = config.get("image", config)
    if not isinstance(caption_config, dict) or not isinstance(image_config, dict):
        raise TypeError("generation_config caption/image entries must be dictionaries")
    text = generate_caption(
        model,
        image,
        prompt=prompt,
        generation_config=caption_config,
    )
    image_prompt = _compose_multimodal_image_prompt(
        prompt=prompt,
        generated_text=text,
        mode=image_prompt_mode,
    )
    generated_image = generate_image(
        model,
        image_prompt,
        generation_config=image_config,
        seed=seed,
        size=size,
    )
    return MultimodalGenerationResult(
        text=text,
        image=generated_image,
        image_prompt=image_prompt,
    )


def _compose_multimodal_image_prompt(*, prompt: str, generated_text: str, mode: str) -> str:
    normalized_mode = mode.strip().lower()
    clean_prompt = " ".join(prompt.split())
    clean_text = " ".join(generated_text.split())
    if normalized_mode == "prompt":
        return clean_prompt
    if normalized_mode == "answer":
        return clean_text or clean_prompt
    if normalized_mode == "prompt-answer":
        if clean_prompt and clean_text:
            return f"{clean_prompt}. {clean_text}"
        return clean_prompt or clean_text
    raise ValueError("image_prompt_mode must be one of: prompt, answer, prompt-answer")


def _predict_guided_noise(
    *,
    model: VlmDiffAdapter,
    latents: torch.Tensor,
    tokens: torch.Tensor,
    unconditional_tokens: torch.Tensor,
    timestep_batch: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    conditional = model(
        {
            "text_tokens": tokens,
            "labels": tokens,
            "image_latents": latents,
            "diffusion_timestep": timestep_batch,
        }
    )["noise_pred"]
    if guidance_scale == 1.0:
        return conditional
    unconditional = model(
        {
            "text_tokens": unconditional_tokens,
            "labels": unconditional_tokens,
            "image_latents": latents,
            "diffusion_timestep": timestep_batch,
        }
    )["noise_pred"]
    return unconditional + guidance_scale * (conditional - unconditional)


def _tokenize_prompt(prompt: str, text_length: int, vocab_size: int) -> torch.Tensor:
    token_ids = torch.zeros(text_length, dtype=torch.long)
    encoded = prompt.encode("utf-8")[:text_length]
    if encoded:
        values = [(byte % (vocab_size - 1)) + 1 for byte in encoded]
        token_ids[: len(values)] = torch.tensor(values, dtype=torch.long)
    return token_ids


def _generate_visual_prefix_caption(
    model: VlmDiffAdapter,
    image: Image.Image,
    prompt: str | None,
    generation_config: dict[str, object],
) -> str:
    text_length = int(generation_config.get("text_length", 64))
    prompt_text = prompt or "Describe the image"
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    image_tensor = _image_to_tensor(image, model.config.image_size * 8).unsqueeze(0).to(device)
    text_tokens = _tokenize_prompt(prompt_text, text_length=text_length, vocab_size=model.vocab_size).unsqueeze(0).to(device)
    with torch.no_grad():
        latents = model.vae.encode(image_tensor)
        batch = {
            "text_tokens": text_tokens,
            "labels": text_tokens,
            "images": image_tensor,
            "image_latents": latents,
            "noise_target": torch.zeros_like(latents),
            "diffusion_timestep": torch.zeros(1, dtype=torch.long, device=device),
        }
        logits = model(batch)["logits"]
        token_ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()
    if was_training:
        model.train()
    return _decode_ascii_tokens(token_ids)


def _generate_causal_visual_prefix_caption(
    model: VlmDiffAdapter,
    image: Image.Image,
    prompt: str | None,
    generation_config: dict[str, object],
) -> str:
    prompt_text = prompt or "Describe the image."
    max_new_tokens = int(generation_config.get("max_new_tokens", 32))
    max_prompt_tokens = int(generation_config.get("max_prompt_tokens", 64))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    image_tensor = _image_to_tensor(image, model.config.image_size * 8).unsqueeze(0).to(device)
    with torch.no_grad():
        latents = model.vae.encode(image_tensor)
        token_ids = model.text_tower.encode(prompt_text, max_length=max_prompt_tokens)
        if not token_ids:
            token_ids = [int(getattr(model.text_tower, "bos_token_id", 1))]
        else:
            token_ids = token_ids + [int(getattr(model.text_tower, "bos_token_id", 1))]
        generated: list[int] = []
        eos_id = int(getattr(model.text_tower, "eos_token_id", -1))
        for _ in range(max_new_tokens):
            tokens = torch.tensor([token_ids + generated], dtype=torch.long, device=device)
            logits = model.causal_logits_from_tokens(
                text_tokens=tokens,
                image_latents=latents,
                images=image_tensor,
            )
            next_id = _select_next_token(
                logits[0, -1],
                prompt_and_generated=token_ids + generated,
                generation_config=generation_config,
            )
            if next_id == eos_id:
                break
            generated.append(next_id)
    if was_training:
        model.train()
    return model.text_tower.decode(generated)


def _select_next_token(
    logits: torch.Tensor,
    *,
    prompt_and_generated: list[int],
    generation_config: dict[str, object],
) -> int:
    adjusted = logits.detach().clone()
    repetition_penalty = float(generation_config.get("repetition_penalty", 1.0))
    if repetition_penalty != 1.0:
        for token_id in set(prompt_and_generated):
            if 0 <= token_id < adjusted.numel():
                if adjusted[token_id] < 0:
                    adjusted[token_id] *= repetition_penalty
                else:
                    adjusted[token_id] /= repetition_penalty
    no_repeat_ngram_size = int(generation_config.get("no_repeat_ngram_size", 0))
    if no_repeat_ngram_size > 0:
        for token_id in _blocked_ngram_tokens(prompt_and_generated, no_repeat_ngram_size):
            if 0 <= token_id < adjusted.numel():
                adjusted[token_id] = -torch.inf
    temperature = float(generation_config.get("temperature", 0.0))
    if temperature <= 0.0:
        return int(adjusted.argmax().item())
    probabilities = torch.softmax(adjusted / temperature, dim=-1)
    top_p = float(generation_config.get("top_p", 1.0))
    if top_p < 1.0:
        probabilities = _top_p_filter(probabilities, top_p=top_p)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def _blocked_ngram_tokens(token_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 1:
        return set(token_ids)
    if len(token_ids) < ngram_size - 1:
        return set()
    prefix = tuple(token_ids[-(ngram_size - 1) :])
    blocked: set[int] = set()
    for index in range(len(token_ids) - ngram_size + 1):
        ngram = tuple(token_ids[index : index + ngram_size])
        if ngram[:-1] == prefix:
            blocked.add(ngram[-1])
    return blocked


def _top_p_filter(probabilities: torch.Tensor, *, top_p: float) -> torch.Tensor:
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    keep = cumulative <= top_p
    keep[0] = True
    filtered = torch.zeros_like(probabilities)
    filtered[sorted_indices[keep]] = sorted_probabilities[keep]
    total = filtered.sum()
    if total <= 0:
        return probabilities
    return filtered / total


def _is_causal_text_tower(model: VlmDiffAdapter) -> bool:
    return hasattr(model.text_tower, "input_embeddings") and hasattr(
        model.text_tower,
        "logits_from_inputs_embeds",
    )


def _image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    values = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).to(torch.float32)
    return values.view(image_size, image_size, 3).permute(2, 0, 1).contiguous() / 127.5 - 1.0


def _decode_ascii_tokens(token_ids: list[int]) -> str:
    decoded = bytes(token for token in token_ids if 32 <= int(token) < 127).decode(
        "ascii",
        errors="ignore",
    )
    return " ".join(decoded.split())


def _tensor_to_image(tensor: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor[:3].clamp(-1.0, 1.0)
    values = ((tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    array = values.permute(1, 2, 0).contiguous().numpy()
    image = Image.fromarray(array, mode="RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image
