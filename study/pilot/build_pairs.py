"""Download verified pilot inputs and construct paired SR crops."""
import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_assets(directory, fetch):
    assets = json.loads((ROOT / "config/assets.json").read_text())
    for relative, (url, digest) in assets.items():
        path = directory / relative
        if not path.is_file() and fetch:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "sr-zero-shot/0.1"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
                if sha(temporary) != digest:
                    raise ValueError(f"Unexpected SHA-256 for {relative}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Missing or invalid input {path}")
    return {name: fields[1] for name, fields in assets.items()}


class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(5):
            setattr(self, f"conv{i + 1}", nn.Conv2d(64 + 32 * i, 32 if i < 4 else 64, 3, padding=1))

    def forward(self, x):
        features = [x]
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
            features.append(F.leaky_relu(layer(torch.cat(features, dim=1)), 0.2))
        return x + 0.2 * self.conv5(torch.cat(features, dim=1))


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = (DenseBlock() for _ in range(3))

    def forward(self, x):
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))


class RealESRGANx2(nn.Module):
    """Inference topology and state keys from BasicSR RRDBNet x2."""
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(12, 64, 3, padding=1)
        self.body = nn.Sequential(*(RRDB() for _ in range(23)))
        self.conv_body = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_hr = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_last = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv_first(F.pixel_unshuffle(x, 2))
        x = x + self.conv_body(self.body(x))
        for layer in (self.conv_up1, self.conv_up2):
            x = F.leaky_relu(layer(F.interpolate(x, scale_factor=2, mode="nearest")), 0.2)
        return self.conv_last(F.leaky_relu(self.conv_hr(x), 0.2))


def load_generators(directory, names):
    result = {}
    for name in names:
        if name == "swinir":
            spec = importlib.util.spec_from_file_location("official_swinir", directory / "network_swinir.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.SwinIR(
                upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.,
                depths=[6] * 4, embed_dim=60, num_heads=[6] * 4, mlp_ratio=2,
                upsampler="pixelshuffledirect", resi_connection="1conv",
            )
            key = "params"
        elif name == "realesrgan":
            model, key = RealESRGANx2(), "params_ema"
        else:
            raise ValueError(f"Unknown generator {name}")
        weights = torch.load(directory / f"{name}.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(weights[key], strict=True)
        result[name] = model.eval()
    return result


def sr_output(model, low, expected_size):
    pixels = np.asarray(low).astype(np.float32) / 255
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        prediction = model(tensor)[0].permute(1, 2, 0)
    if prediction.shape != (expected_size, expected_size, 3) or not torch.isfinite(prediction).all():
        raise ValueError("Unexpected SR prediction shape or values")
    output = prediction.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(output)


def color_normalize(image):
    image = ImageOps.exif_transpose(image)
    profile = image.info.get("icc_profile")
    if profile:
        image = ImageCms.profileToProfile(
            image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
            ImageCms.createProfile("sRGB"), outputMode="RGB",
        )
    return image.convert("RGB")


def write_pairs(directory, output, cfg, asset_hashes):
    torch.set_num_threads(cfg["torch_threads"])
    names, context, size = cfg["generator_ids"], cfg["context_px"], cfg["input_px"]
    if cfg["scale"] != 2 or context % 16 or (context - size) % 4 or context < size:
        raise ValueError("Use a context divisible by 16 and an even margin per side")
    generators = load_generators(directory / "models", names)
    manifest, source_info, seen_source_pixels, seen_reference_pixels = [], [], set(), set()
    with zipfile.ZipFile(directory / "DIV2K_valid_HR.zip") as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Invalid archive member CRC {bad_member}")
        members = sorted(x for x in archive.namelist() if x.lower().endswith(".png"))
        rng = np.random.default_rng(cfg["seed"])
        counts = cfg["split_sources"]
        if sum(counts) != len(members) or min(counts) < 1 or len(counts) != 3:
            raise ValueError("Source counts must cover the archive exactly")
        order = rng.permutation(members)
        splits = dict(zip(order.tolist(), np.repeat(["train", "val", "test"], counts).tolist()))
        margin = (context - size) // 2
        inner = (margin, margin, margin + size, margin + size)
        for index, member in enumerate(members, 1):
            source_id = Path(member).stem
            raw = archive.read(member)
            with Image.open(io.BytesIO(raw)) as decoded:
                source = color_normalize(decoded)
                source_mode, had_icc = decoded.mode, "icc_profile" in decoded.info
            width, height = source.size
            if min(width, height) < context:
                raise ValueError(f"Source too small {member}")
            source_pixels = hashlib.sha256(source.tobytes()).hexdigest()
            if source_pixels in seen_source_pixels:
                raise ValueError(f"Duplicate decoded source {member}")
            seen_source_pixels.add(source_pixels)
            left, top = (width - context) // 2, (height - context) // 2
            original = source.crop((left, top, left + context, top + context))
            low = original.resize((context // 2,) * 2, Image.Resampling.BICUBIC)
            variants = {"original": original, "bicubic": low.resize(original.size, Image.Resampling.BICUBIC)}
            variants.update({name: sr_output(model, low, context) for name, model in generators.items()})
            reference_digest = hashlib.sha256(original.crop(inner).tobytes()).hexdigest()
            if reference_digest in seen_reference_pixels:
                raise ValueError(f"Duplicate reference crop {member}")
            seen_reference_pixels.add(reference_digest)
            if len({hashlib.sha256(im.crop(inner).tobytes()).hexdigest()
                    for im in variants.values()}) != len(variants):
                raise ValueError(f"Duplicate variants {member}")
            source_info.append(dict(source_id=source_id, archive_member=member, split=splits[member],
                                    width=width, height=height, mode=source_mode, icc=had_icc,
                                    source_sha256=hashlib.sha256(raw).hexdigest(),
                                    reference_pixel_sha256=reference_digest, context_xy=[left, top]))
            split_dir = output / splits[member]
            split_dir.mkdir(exist_ok=True)
            for name, image in {"lr": low, **variants}.items():
                if name != "lr":
                    image = image.crop(inner)
                path = split_dir / f"{source_id}_{name}.png"
                # A fresh image removes inherited EXIF, ICC, and PNG text chunks.
                Image.frombytes("RGB", image.size, image.tobytes()).save(path, compress_level=6)
                with Image.open(path) as saved:
                    if saved.mode != "RGB" or saved.format != "PNG" or saved.size != image.size or saved.info:
                        raise ValueError(f"Output encoding mismatch {path}")
                    saved.load()
                manifest.append(dict(source_id=source_id, split=splits[member], variant=name,
                                     path=str(path.relative_to(output)), sha256=sha(path),
                                     width=image.width, height=image.height))
            if index % 20 == 0:
                print(f"Built {index}/{len(members)} sources", flush=True)
    inventory = dict(assets_sha256=asset_hashes, config=cfg,
                     source_counts=Counter(row["split"] for row in source_info),
                     item_counts=Counter(row["variant"] for row in manifest), sources=source_info)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/pilot.json")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    assets = verify_assets(args.assets, args.fetch)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        staging = Path(temporary) / "pairs"
        staging.mkdir()
        write_pairs(args.assets, staging, cfg, assets)
        staging.rename(args.out)


if __name__ == "__main__":
    main()
