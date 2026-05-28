import json
from pathlib import Path

import torch
from PIL import Image

from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.data import build_manifest_batch
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_manifest_batch_uses_noised_latents_and_true_noise_target(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    manifest = tmp_path / "manifest.jsonl"
    Image.new("RGB", (48, 48), color="purple").save(image_path)
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "image_path": str(image_path),
                "caption": "purple square",
                "clip_score": 0.9,
            }
        ),
        encoding="utf-8",
    )
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))

    torch.manual_seed(123)
    batch = build_manifest_batch(
        model=model,
        manifest_path=manifest,
        batch_size=1,
        text_length=6,
        device=torch.device("cpu"),
    ).tensors

    assert set(["clean_latents", "image_latents", "noise_target", "diffusion_timestep"]).issubset(batch)
    assert batch["clean_latents"].shape == batch["image_latents"].shape
    assert batch["noise_target"].shape == batch["image_latents"].shape
    assert batch["diffusion_timestep"].shape == (1,)
    assert batch["diffusion_timestep"].dtype == torch.long
    assert batch["noise_target"].abs().sum().item() > 0
    assert not torch.allclose(batch["image_latents"], batch["clean_latents"])
