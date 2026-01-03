import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

_SAVERS: dict[str, Callable] = {}


def register_saver(*fmts: str):
    """Register a saver under one or multiple format aliases."""
    norm = [f.lower().lstrip(".") for f in fmts]

    def deco(func: Callable) -> Callable:
        for f in norm:
            if f in _SAVERS:
                raise ValueError(f"Duplicate saver for format: {f}")
            _SAVERS[f] = func
        return func

    return deco


def write_records(records: dict[str, list[Any]] | list[Any], path: Path | str, **kwargs) -> None:
    """Write records to disk using the registered saver for the given format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".").lower()
    if fmt == "":
        raise ValueError("Output path must have a file extension indicating the format.")
    elif fmt not in _SAVERS:
        raise ValueError(f"No saver registered for format: {fmt}")
    _SAVERS[fmt](records=records, path=path, **kwargs)


# ---------- JSON ----------
@register_saver("json")
def write_json(*, records: dict[str, list[Any]], path: Path, ensure_ascii=False) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=ensure_ascii, indent=2)


# ---------- Pickle ----------
@register_saver("pkl", "pickle")
def write_pickle(*, records: dict[str, list[Any]], path: Path) -> None:
    import pickle

    with path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------- NPY ----------
@register_saver("numpy", "npy")
def write_npy(*, records: dict[str, list[Any]], path: Path) -> None:
    np.save(path, np.array(records, dtype=object), allow_pickle=True)


# ---------- Parquet ----------
@register_saver("parquet")
def write_parquet(*, records: dict[str, list[Any]], path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


# ---------- Video ----------
@register_saver("video", "mp4")
def write_video_imageio(*, records: list[np.ndarray], path: Path, video_fps=30) -> None:
    import imageio.v3 as iio
    import numpy as np

    first = records[0]
    if not isinstance(first, np.ndarray) or first.ndim != 3 or first.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported frame shape for video: {getattr(first, 'shape', None)}")

    h, w = first.shape[:2]

    frames = []
    for frame in records:
        if frame.shape[:2] != (h, w):
            raise ValueError("All frames must have the same shape for video output.")

        rgb = frame[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        frames.append(rgb)

    iio.imwrite(uri=path, image=frames, fps=video_fps, codec="libx264", pixelformat="yuv420p")
