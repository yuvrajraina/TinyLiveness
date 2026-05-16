from __future__ import annotations

import csv
import random
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
LIVE_NAMES = {"bona_fide", "bona-fide", "genuine", "live", "real"}
SPOOF_NAMES = {"attack", "fake", "mask", "photo", "print", "replay", "screen", "spoof", "video"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
AUGMENTATION_PRESETS: dict[str, dict[str, float | tuple[float, float] | tuple[int, int]]] = {
    "light": {
        "flip": 0.5,
        "brightness": (0.90, 1.10),
        "contrast": (0.90, 1.12),
        "color": (0.92, 1.10),
        "gamma": (0.90, 1.12),
        "blur": 0.10,
        "jpeg": 0.10,
        "jpeg_quality": (65, 95),
        "downscale": 0.05,
        "noise": 0.05,
        "cutout": 0.04,
    },
    "medium": {
        "flip": 0.5,
        "brightness": (0.75, 1.25),
        "contrast": (0.75, 1.30),
        "color": (0.75, 1.25),
        "gamma": (0.75, 1.35),
        "blur": 0.25,
        "motion_blur": 0.08,
        "jpeg": 0.25,
        "jpeg_quality": (35, 92),
        "downscale": 0.15,
        "noise": 0.15,
        "cutout": 0.10,
        "sharpen": 0.08,
        "temperature": 0.15,
        "resized_crop": 0.10,
    },
    "strong": {
        "flip": 0.5,
        "brightness": (0.60, 1.40),
        "contrast": (0.60, 1.50),
        "color": (0.65, 1.40),
        "gamma": (0.60, 1.60),
        "blur": 0.35,
        "motion_blur": 0.15,
        "jpeg": 0.35,
        "jpeg_quality": (25, 88),
        "downscale": 0.30,
        "noise": 0.25,
        "cutout": 0.16,
        "sharpen": 0.18,
        "temperature": 0.25,
        "moire": 0.12,
        "paper_noise": 0.12,
        "resized_crop": 0.18,
    },
    "spoof_artifact_strong": {
        "flip": 0.5,
        "brightness": (0.55, 1.45),
        "contrast": (0.55, 1.60),
        "color": (0.60, 1.45),
        "gamma": (0.55, 1.70),
        "blur": 0.40,
        "motion_blur": 0.20,
        "jpeg": 0.55,
        "jpeg_quality": (18, 82),
        "downscale": 0.45,
        "noise": 0.35,
        "cutout": 0.18,
        "sharpen": 0.25,
        "temperature": 0.30,
        "moire": 0.25,
        "paper_noise": 0.22,
        "resized_crop": 0.20,
    },
}


@dataclass(frozen=True)
class LivenessSample:
    path: Path
    label: int

    @property
    def label_name(self) -> str:
        return "live" if self.label == 1 else "spoof"


class LivenessImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Loads liveness crops from folder or CSV manifests.

    Folder layout:

    ``split/live/*.jpg`` and ``split/spoof/*.jpg`` where ``split`` is usually
    ``train`` or ``val``. Aliases such as ``real``/``fake`` and
    ``bona_fide``/``attack`` are accepted.

    CSV layout:

    ``path,label`` with label values ``live``, ``spoof``, ``1`` or ``0``.
    Relative paths are resolved from the CSV file's parent directory.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        image_size: int = 112,
        augment: bool = False,
        repeats: int = 1,
        normalization: str = "tinyliveness",
        augmentation_preset: str = "medium",
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset path not found: {self.root}")
        if repeats < 1:
            raise ValueError("repeats must be >= 1")

        self.image_size = image_size
        self.augment = augment
        self.repeats = repeats
        self.normalization = normalization
        self.augmentation_preset = augmentation_preset

        if self.root.is_file():
            self.samples = self._load_csv(self.root)
        else:
            self.samples = self._load_folder(self.root)
        if not self.samples:
            raise ValueError(f"no liveness samples found in {self.root}")

        labels = [sample.label for sample in self.samples]
        if len(set(labels)) < 2:
            raise ValueError("liveness dataset must contain both live and spoof samples")

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index % len(self.samples)]
        image = load_rgb_image(sample.path, self.image_size)
        augment_tensor = False
        if self.augment:
            image = augment_liveness_image(
                image,
                preset=self.augmentation_preset,
                label_name=sample.label_name,
            )
            augment_tensor = True
        image_tensor = image_to_tensor(
            image,
            augment=augment_tensor,
            normalization=self.normalization,
        )
        label_tensor = torch.tensor([sample.label], dtype=torch.float32)
        return image_tensor, label_tensor

    def class_counts(self) -> dict[str, int]:
        live = sum(sample.label == 1 for sample in self.samples)
        spoof = len(self.samples) - live
        return {"live": live, "spoof": spoof}

    def add_samples(self, samples: list[LivenessSample]) -> None:
        self.samples.extend(samples)

    @staticmethod
    def _load_folder(root: Path) -> list[LivenessSample]:
        samples: list[LivenessSample] = []
        for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            label = label_from_name(label_dir.name)
            if label is None:
                continue
            for image_path in sorted(label_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(LivenessSample(path=image_path, label=label))
        return samples

    @staticmethod
    def _load_csv(path: Path) -> list[LivenessSample]:
        samples: list[LivenessSample] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing columns: {sorted(missing)}")
            for row in reader:
                label = parse_label(row["label"])
                image_path = Path(row["path"])
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                samples.append(LivenessSample(path=image_path, label=label))
        return samples


def label_from_name(name: str) -> int | None:
    normalized = name.strip().lower().replace(" ", "_")
    if normalized in LIVE_NAMES:
        return 1
    if normalized in SPOOF_NAMES:
        return 0
    return None


def parse_label(label: str) -> int:
    normalized = label.strip().lower().replace(" ", "_")
    if normalized in LIVE_NAMES or normalized in {"1", "true"}:
        return 1
    if normalized in SPOOF_NAMES or normalized in {"0", "false"}:
        return 0
    raise ValueError(f"unknown liveness label: {label!r}")


def load_rgb_image(path: Path, image_size: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)


def get_augmentation_config(name: str) -> dict[str, object]:
    if name not in AUGMENTATION_PRESETS:
        raise ValueError(f"unknown augmentation preset: {name!r}")
    return dict(AUGMENTATION_PRESETS[name])


def augment_liveness_image(
    image: Image.Image,
    *,
    preset: str = "medium",
    label_name: str = "",
) -> Image.Image:
    if preset == "none":
        return image
    config = get_augmentation_config(preset)
    if preset == "spoof_artifact_strong" and label_name == "live":
        config = get_augmentation_config("medium")

    if random.random() < float(config.get("flip", 0.0)):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < float(config.get("resized_crop", 0.0)):
        image = random_resized_crop(image, min_scale=0.84, max_scale=1.0)

    if "brightness" in config:
        image = ImageEnhance.Brightness(image).enhance(random_range(config["brightness"]))
    if "contrast" in config:
        image = ImageEnhance.Contrast(image).enhance(random_range(config["contrast"]))
    if "color" in config:
        image = ImageEnhance.Color(image).enhance(random_range(config["color"]))
    if "gamma" in config:
        image = apply_gamma(image, random_range(config["gamma"]))

    if random.random() < float(config.get("temperature", 0.0)):
        image = color_temperature_shift(image, strength=random.uniform(-0.12, 0.12))

    if random.random() < float(config.get("blur", 0.0)):
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))

    if random.random() < float(config.get("motion_blur", 0.0)):
        image = motion_blur(image)

    if random.random() < float(config.get("jpeg", 0.0)):
        quality = config.get("jpeg_quality", (35, 92))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*quality))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert("RGB")

    if random.random() < float(config.get("downscale", 0.0)):
        width, height = image.size
        scale = random.uniform(0.45, 0.90)
        small = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
        image = small.resize((width, height), Image.BILINEAR)

    if random.random() < float(config.get("moire", 0.0)):
        image = add_moire(image)
    if random.random() < float(config.get("paper_noise", 0.0)):
        image = add_paper_noise(image)
    if random.random() < float(config.get("sharpen", 0.0)):
        image = ImageEnhance.Sharpness(image).enhance(random.uniform(1.5, 3.5))

    return image


def random_range(value: object) -> float:
    lower, upper = value
    return random.uniform(float(lower), float(upper))


def random_resized_crop(image: Image.Image, *, min_scale: float, max_scale: float) -> Image.Image:
    width, height = image.size
    scale = random.uniform(min_scale, max_scale)
    crop_w = max(1, int(width * scale))
    crop_h = max(1, int(height * scale))
    left = random.randint(0, width - crop_w)
    top = random.randint(0, height - crop_h)
    return image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.BILINEAR)


def apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    inverse = 1.0 / max(gamma, 1e-6)
    table = [min(255, max(0, int(((value / 255.0) ** inverse) * 255.0))) for value in range(256)]
    return image.point(table * 3)


def color_temperature_shift(image: Image.Image, *, strength: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32).copy()
    array[..., 0] *= 1.0 + strength
    array[..., 2] *= 1.0 - strength
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def motion_blur(image: Image.Image) -> Image.Image:
    size = random.choice([3, 5])
    kernel = np.zeros((size, size), dtype=np.float32)
    if random.random() < 0.5:
        kernel[size // 2, :] = 1.0 / size
    else:
        kernel[:, size // 2] = 1.0 / size
    return image.filter(ImageFilter.Kernel((size, size), kernel.reshape(-1).tolist(), scale=1.0))


def add_moire(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    height, width, _ = array.shape
    yy, xx = np.mgrid[0:height, 0:width]
    period = random.uniform(5.0, 12.0)
    phase = random.uniform(0.0, 2.0 * np.pi)
    pattern = np.sin((xx + yy) / period + phase) * random.uniform(4.0, 12.0)
    array = array + pattern[..., None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def add_paper_noise(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=random.uniform(3.0, 9.0), size=array.shape)
    grain = np.random.uniform(0.92, 1.08, size=(array.shape[0], array.shape[1], 1))
    array = array * grain + noise
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def image_to_tensor(
    image: Image.Image,
    *,
    augment: bool = False,
    normalization: str = "tinyliveness",
) -> Tensor:
    array = np.asarray(image, dtype=np.float32)
    if normalization == "tinyliveness":
        normalized = (array - 127.5) / 128.0
    elif normalization == "imagenet":
        normalized = ((array / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    else:
        raise ValueError(f"unknown normalization: {normalization!r}")
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).contiguous()

    if augment and random.random() < 0.15:
        noise = torch.randn_like(tensor) * 0.015
        tensor = tensor + noise
    if augment and random.random() < 0.10:
        channels, height, width = tensor.shape
        erase_h = int(torch.randint(8, 22, ()).item())
        erase_w = int(torch.randint(8, 22, ()).item())
        top = int(torch.randint(0, height - erase_h + 1, ()).item())
        left = int(torch.randint(0, width - erase_w + 1, ()).item())
        tensor = tensor.clone()
        tensor[:, top : top + erase_h, left : left + erase_w] = 0.0

    if normalization == "tinyliveness":
        return tensor.clamp_(-1.0, 1.0)
    return tensor
