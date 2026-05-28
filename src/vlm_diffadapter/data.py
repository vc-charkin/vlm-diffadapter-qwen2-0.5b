from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor
from PIL import Image

from vlm_diffadapter.config import DataConfig
from vlm_diffadapter.diffusion import add_diffusion_noise, sample_diffusion_timesteps

if TYPE_CHECKING:
    from vlm_diffadapter.modeling import VlmDiffAdapter

TaskName = Literal["t2i", "i2t"]


@dataclass(frozen=True)
class NoisePolicy:
    timestep: int
    add_noise: bool


@dataclass(frozen=True)
class ManifestBatch:
    tensors: dict[str, Tensor]
    sample_ids: list[str]
    manifest_size: int


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    image_path: Path
    caption: str
    clip_score: float


def apply_noise_policy(task: TaskName, clean_i2t: bool, timestep: int) -> NoisePolicy:
    if task == "i2t" and clean_i2t:
        return NoisePolicy(timestep=0, add_noise=False)
    return NoisePolicy(timestep=timestep, add_noise=True)


class TaskRatioSampler:
    def __init__(self, config: DataConfig) -> None:
        left, right = config.t2i_i2t_ratio.split(":", maxsplit=1)
        self.pattern: list[TaskName] = ["t2i"] * int(left) + ["i2t"] * int(right)

    def task_for_index(self, index: int) -> TaskName:
        return self.pattern[index % len(self.pattern)]


class ManifestDataModule:
    def __init__(
        self,
        train_records: list[ManifestRecord],
        val_records: list[ManifestRecord],
        report: dict[str, int],
    ) -> None:
        self.train_records = list(train_records)
        self.val_records = list(val_records)
        self.report = dict(report)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        config: DataConfig,
        val_fraction: float = 0.0,
    ) -> ManifestDataModule:
        raw_records = read_jsonl(manifest_path)
        kept: list[ManifestRecord] = []
        report = {
            "total": len(raw_records),
            "kept": 0,
            "filtered_clip_score": 0,
            "filtered_short_caption": 0,
            "broken_images": 0,
            "train": 0,
            "val": 0,
            "seed": config.seed,
        }
        for raw in raw_records:
            caption = str(raw.get("caption", ""))
            if float(raw.get("clip_score", 1.0)) < config.clip_threshold:
                report["filtered_clip_score"] += 1
                continue
            if len(caption) < config.min_caption_chars:
                report["filtered_short_caption"] += 1
                continue
            image_path = Path(str(raw["image_path"]))
            if not _is_valid_image(image_path):
                report["broken_images"] += 1
                continue
            kept.append(
                ManifestRecord(
                    sample_id=str(raw.get("id", image_path.stem)),
                    image_path=image_path,
                    caption=caption,
                    clip_score=float(raw.get("clip_score", 1.0)),
                )
            )

        shuffled = list(kept)
        random.Random(config.seed).shuffle(shuffled)
        val_count = _split_count(len(shuffled), val_fraction)
        val_records = shuffled[:val_count]
        train_records = shuffled[val_count:]
        report["kept"] = len(kept)
        report["train"] = len(train_records)
        report["val"] = len(val_records)
        return cls(train_records=train_records, val_records=val_records, report=report)

    @property
    def train_ids(self) -> list[str]:
        return [record.sample_id for record in self.train_records]

    @property
    def val_ids(self) -> list[str]:
        return [record.sample_id for record in self.val_records]

    def iter_split_batches(
        self,
        split: Literal["train", "val"],
        batch_size: int,
        drop_last: bool = False,
    ) -> list[list[ManifestRecord]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        records = self.train_records if split == "train" else self.val_records
        batches: list[list[ManifestRecord]] = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            if drop_last and len(batch) < batch_size:
                continue
            batches.append(batch)
        return batches


def _split_count(total: int, fraction: float) -> int:
    if fraction <= 0 or total == 0:
        return 0
    if fraction >= 1:
        return total
    return max(1, int(total * fraction))


def _is_valid_image(path: str | Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def prepare_manifest(
    records: list[dict[str, Any]],
    clip_threshold: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    del seed
    kept: list[dict[str, Any]] = []
    report = {"kept": 0, "filtered_clip_score": 0, "broken_images": 0}

    for record in records:
        if float(record.get("clip_score", 1.0)) < clip_threshold:
            report["filtered_clip_score"] += 1
            continue
        if not _is_valid_image(record["image_path"]):
            report["broken_images"] += 1
            continue
        kept.append(dict(record))

    report["kept"] = len(kept)
    return kept, report


def build_manifest_batch(
    model: VlmDiffAdapter,
    manifest_path: str | Path,
    batch_size: int,
    text_length: int,
    device: torch.device,
) -> ManifestBatch:
    records = read_jsonl(manifest_path)
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    selected = [records[index % len(records)] for index in range(batch_size)]
    images = torch.stack(
        [_load_image_tensor(record["image_path"], model.config.image_size * 8) for record in selected],
    ).to(device)
    text_tokens = torch.stack(
        [
            _tokenize_caption(
                caption=str(record.get("caption", "")),
                text_length=text_length,
                vocab_size=model.vocab_size,
            )
            for record in selected
        ],
    ).to(device)
    with torch.no_grad():
        clean_latents = model.vae.encode(images)
    diffusion = _build_diffusion_training_tensors(
        clean_latents,
        schedule=model.config.diffusion_schedule,
    )
    batch = {
        "text_tokens": text_tokens,
        "labels": text_tokens.clone(),
        "images": images,
        **diffusion,
    }
    return ManifestBatch(
        tensors=batch,
        sample_ids=[str(record.get("id", Path(str(record["image_path"])).stem)) for record in selected],
        manifest_size=len(records),
    )


def build_manifest_batch_from_records(
    model: VlmDiffAdapter,
    records: list[ManifestRecord],
    manifest_size: int,
    text_length: int,
    device: torch.device,
) -> ManifestBatch:
    if not records:
        raise ValueError("records must not be empty")
    images = torch.stack(
        [_load_image_tensor(record.image_path, model.config.image_size * 8) for record in records],
    ).to(device)
    text_tokens = torch.stack(
        [
            _tokenize_caption(
                caption=record.caption,
                text_length=text_length,
                vocab_size=model.vocab_size,
            )
            for record in records
        ],
    ).to(device)
    with torch.no_grad():
        clean_latents = model.vae.encode(images)
    diffusion = _build_diffusion_training_tensors(
        clean_latents,
        schedule=model.config.diffusion_schedule,
    )
    batch = {
        "text_tokens": text_tokens,
        "labels": text_tokens.clone(),
        "images": images,
        **diffusion,
    }
    return ManifestBatch(
        tensors=batch,
        sample_ids=[record.sample_id for record in records],
        manifest_size=manifest_size,
    )


def _load_image_tensor(path: str | Path, image_size: int) -> Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        values = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).to(torch.float32)
    return values.view(image_size, image_size, 3).permute(2, 0, 1).contiguous() / 127.5 - 1.0


def _build_diffusion_training_tensors(clean_latents: Tensor, schedule: str = "linear") -> dict[str, Tensor]:
    noise = torch.randn_like(clean_latents)
    timesteps = sample_diffusion_timesteps(clean_latents.shape[0], clean_latents.device)
    noised_latents = add_diffusion_noise(clean_latents, noise, timesteps, schedule=schedule)
    return {
        "clean_latents": clean_latents,
        "image_latents": noised_latents,
        "noise_target": noise,
        "diffusion_timestep": timesteps,
    }


def _tokenize_caption(caption: str, text_length: int, vocab_size: int) -> Tensor:
    token_ids = torch.zeros(text_length, dtype=torch.long)
    encoded = caption.encode("utf-8")[:text_length]
    if encoded:
        values = [(byte % (vocab_size - 1)) + 1 for byte in encoded]
        token_ids[: len(values)] = torch.tensor(values, dtype=torch.long)
    return token_ids


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                loaded = json.loads(stripped)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                records.append(loaded)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    return output_path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
