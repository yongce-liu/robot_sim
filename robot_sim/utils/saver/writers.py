import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

_WRITER_REGISTRY: dict[str, Callable] = {}


def writer_register(*fmts: str):
    """Register a saver under one or multiple format aliases."""
    norm = [f.lower().lstrip(".") for f in fmts]

    def deco(func: Callable) -> Callable:
        for f in norm:
            if f in _WRITER_REGISTRY:
                raise ValueError(f"Duplicate saver for format: {f}")
            _WRITER_REGISTRY[f] = func
        return func

    return deco


def write_records(records: dict[str, Any] | list[Any], path: Path | str, **kwargs) -> None:
    """Write records to disk using the registered saver for the given format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".").lower()
    if fmt == "":
        raise ValueError("Output path must have a file extension indicating the format.")
    elif fmt not in _WRITER_REGISTRY:
        raise ValueError(f"No saver registered for format: {fmt}")
    _WRITER_REGISTRY[fmt](records=records, path=path, **kwargs)


# ---------- JSON ----------
@writer_register("json")
def write_json(*, records: dict[str, list[Any]], path: Path, ensure_ascii=False) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=ensure_ascii, indent=2)


# ---------- JSONl ----------
@writer_register("jsonl")
def write_jsonl(*, records: Iterable[Any], path: Path, ensure_ascii: bool = False) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=ensure_ascii) + "\n")


# ---------- Pickle ----------
@writer_register("pkl", "pickle")
def write_pickle(*, records: dict[str, list[Any]], path: Path) -> None:
    import pickle

    with path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------- NPY ----------
@writer_register("numpy", "npy")
def write_npy(*, records: dict[str, list[Any]], path: Path) -> None:
    np.save(path, np.array(records, dtype=object), allow_pickle=True)


# ---------- Parquet ----------
@writer_register("parquet")
def write_parquet(*, records: dict[str, list[Any]], path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


# ---------- Video ----------
@writer_register("video", "mp4")
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
