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


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_records(records: list[dict[str, Any]], path: Path | str, **kwargs) -> None:
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
def write_json(*, records: list[dict[str, Any]], path: Path, ensure_ascii=False) -> None:
    payload = [to_jsonable(r) for r in records]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=ensure_ascii, indent=2)


# ---------- Pickle ----------
@register_saver("pkl", "pickle")
def write_pickle(*, records: list[dict[str, Any]], path: Path) -> None:
    import pickle

    with path.open("wb") as f:
        pickle.dump(list(records), f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------- NPY ----------
@register_saver("npy")
def write_npy(*, records: list[dict[str, Any]], path: Path) -> None:
    np.save(path, np.array(list(records), dtype=object), allow_pickle=True)


# ---------- Parquet ----------
@register_saver("parquet")
def write_parquet(*, records: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    payload = [to_jsonable(r) for r in records]
    df = pd.DataFrame(payload)
    df.to_parquet(path, index=False)


# ---------- TensorDict ----------
@register_saver("tensordict", "td", "pt")
def write_tensordict(*, records: list[dict[str, Any]], path: Path) -> None:
    import torch
    from tensordict import TensorDict

    recs = list(records)
    n = len(recs)

    def to_tensor(v: Any):
        # torch.Tensor or tensor-like
        if hasattr(v, "detach") and hasattr(v, "cpu"):
            v = v.detach().cpu()
            return v if hasattr(v, "shape") else None
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        if isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating)):
            return torch.tensor(v)
        if (
            isinstance(v, (list, tuple))
            and v
            and all(isinstance(x, (bool, int, float, np.bool_, np.integer, np.floating)) for x in v)
        ):
            return torch.tensor(v)
        return None

    def to_td_or_tensor(v: Any):
        if isinstance(v, dict):
            converted = {}
            for k, vv in v.items():
                inner = to_td_or_tensor(vv)
                if inner is None:
                    return None
                converted[str(k)] = inner
            return TensorDict(converted, batch_size=[])
        return to_tensor(v)

    def stack(values: list[Any]):
        converted = [to_td_or_tensor(v) for v in values]
        if any(x is None for x in converted):
            return None
        if all(isinstance(x, TensorDict) for x in converted):
            return TensorDict.stack(converted, dim=0)
        if all(torch.is_tensor(x) for x in converted):
            return torch.stack(converted, dim=0)
        return None

    tensor_data: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    tensor_data["episode"] = torch.tensor([int(r.get("episode", 0)) for r in recs], dtype=torch.int64)
    tensor_data["step"] = torch.tensor([int(r.get("step", 0)) for r in recs], dtype=torch.int64)

    for key in ["reward", "terminated", "truncated", "observation", "action"]:
        values = [r.get(key, None) for r in recs]
        stacked = stack(values)
        if stacked is None:
            metadata[key] = [to_jsonable(v) for v in values]
        else:
            tensor_data[key] = stacked

    for key in ["info", "event", "timestamp", "render"]:
        values = [r.get(key, None) for r in recs]
        if any(v is not None for v in values):
            metadata[key] = [to_jsonable(v) for v in values]

    payload = {
        "data": TensorDict(tensor_data, batch_size=[n]),
        "metadata": metadata,
    }
    torch.save(payload, path)


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
