from collections import defaultdict
from pathlib import Path
from typing import Any

from robot_sim.utils.saver import SingleRecord, write_records

from .env import Gr00tEnv


class Gr00tDatasetsBuilder:
    def __init__(self, env: Gr00tEnv, source: dict[int, list[SingleRecord]] | None = None):
        self.step_dt = env.step_dt
        self.src: dict[int, list[SingleRecord]] | None = source  # episode id: data
        self.obs_config: dict[str, Any] = env._map_config["observation"]
        self.act_config: dict[str, Any] = env._map_config["action"]

    def init_spec(
        self,
        source: dict[int, list[SingleRecord]],
        output_dir: str | Path = "outputs",
        chunk_size: int = 1000,
        chunk_key="episode_chunk",
        episode_key="episode_index",
    ):
        self.src = source
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size: int = chunk_size
        self.chunk_key: str = chunk_key
        self.episode_key: str = episode_key
        self.chunk_format = f"chunk-{{{chunk_key}:03d}}"
        self.episode_format = f"episode_{{{episode_key}:06d}}"

    def build_dataset(self, source: dict[int, list[SingleRecord]], **kwargs):
        self.init_spec(source=source, **kwargs)
        self.output_data()
        self.output_video()
        self.output_meta()

    def output_meta(self):
        self._build_episodes()
        self._build_info()
        self._build_modality()
        self._build_tasks()

    def output_data(self, suffix: str = "parquet"):
        assert self.src is not None, "Source Data Should be initialized"
        output_format = f"{self.output_dir}/data/{self.chunk_format}/{self.episode_format}.{suffix.lstrip('.')}"
        self.output_data_format = output_format
        avai_obs_keys = [k for k, v in self.obs_config.items() if v.get("type") in ["joint", "command"]]
        avai_act_keys = [k for k, v in self.act_config.items() if v.get("type") in ["joint", "command"]]
        for ep_idx, records in self.src.items():
            output_name = output_format.format(
                **{self.chunk_key: ep_idx // self.chunk_size, self.episode_key: ep_idx % self.chunk_size}
            )
            data = []
            for _re in records:
                piece = {
                    "timestamp": _re.timestamp,
                    "frame_index": _re.step,
                    "episode_index": ep_idx,
                    "index": _re.step,
                    "task_index": 1,
                }
                for k in avai_obs_keys:
                    piece[k] = _re.observation[k] if _re.observation is not None else None
                for k in avai_act_keys:
                    piece[k] = _re.action[k].reshape(-1) if _re.action is not None else None
                data.append(piece)
            write_records(data, path=output_name)

    def output_video(self, suffix: str = "mp4"):
        assert self.src is not None, "Source Data Should be initialized"
        output_format = (
            f"{self.output_dir}/videos/{self.chunk_format}/{{video_key}}/{self.episode_format}.{suffix.lstrip('.')}"
        )
        self.output_video_format = output_format
        video_keys = [k for k, v in self.obs_config.items() if v.get("type") == "sensor"]
        for video_key in video_keys:
            for ep_idx, records in self.src.items():
                output_name = output_format.format(
                    **{
                        self.chunk_key: ep_idx // self.chunk_size,
                        "video_key": video_key,
                        self.episode_key: ep_idx % self.chunk_size,
                    }
                )
                write_records(
                    [item.observation[video_key] for item in records], path=output_name, video_fps=int(1 / self.step_dt)
                )

    def _build_episodes(self, suffix: str = "jsonl"):
        assert self.src is not None, "Source Data Should be initialized"
        output_name = f"{self.output_dir}/meta/episodes.{suffix.lstrip('.')}"
        data: list[dict] = []
        for ep_idx, records in self.src.items():
            data.append(
                {
                    "episode_index": ep_idx,
                    "tasks": [
                        v.get("value") for v in self.obs_config.values() if v.get("type") in ["constant", "task"]
                    ],
                    "length": len(records),
                    "discarded_trajectory": False,
                    "trajectory_type": "successful",
                }
            )
        write_records(data, output_name)

    def _build_info(self, suffix: str = "json"):
        assert self.src is not None, "Source Data Should be initialized"
        output_name = f"{self.output_dir}/meta/info.{suffix.lstrip('.')}"

        data = {
            "total_episodes": len(self.src),
            "total_frames": sum([len(v) for v in self.src.values()]),
            "total_tasks": 1 + len([1 for v in self.obs_config.values() if v.get("type") in ["constant", "task"]]),
            "total_videos": len(self.src),
            "total_chunks": (len(self.src) + self.chunk_size - 1) // self.chunk_size,
            "chunks_size": self.chunk_size,
            "fps": int(1 / self.step_dt),
            "splits": {"train": "0:100"},
            "data_path": f"data/{self.chunk_format}/{self.episode_format}.parquet",
            "video_path": f"videos/{self.chunk_format}/{{video_key}}/{self.episode_format}.mp4",
        }

        features = {
            k: {
                "dtype": str(self.src[next(iter(self.src))][-1].observation[k].dtype),
                "shape": self.src[next(iter(self.src))][-1].observation[k].shape,
            }
            for k, cfg in self.obs_config.items()
            if cfg.get("type") in ["joint", "command"]
        }
        features.update(
            {
                k: {
                    "dtype": str(self.src[next(iter(self.src))][-1].action[k].dtype),
                    "shape": self.src[next(iter(self.src))][-1].action[k].reshape(-1).shape,
                }
                for k, cfg in self.act_config.items()
                if cfg.get("type") in ["joint", "command"]
            }
        )
        data.update({"features": features})

        write_records(data, output_name)

    def _build_modality(self, suffix: str = "json"):
        assert self.src is not None, "Source Data Should be initialized"
        output_name = f"{self.output_dir}/meta/modality.{suffix.lstrip('.')}"
        data: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        for config in [self.obs_config, self.act_config]:
            for k, v in config.items():
                group_name, part_name = k.split(".", 1)
                if v.get("type") in ["joint", "command"]:
                    if v.get("indices"):
                        length = len(v["indices"])
                    elif v.get("bound"):
                        length = len(v["bound"]["min"])
                    else:
                        raise ValueError
                    data[group_name][part_name] = {
                        "original_key": k,
                        "start": 0,
                        "end": length,
                        "rotation_type": None,
                        "absolute": True,
                        "dtype": "float64",
                        "range": None,
                    }
                elif v.get("type") in ["sensor"]:
                    data[group_name][part_name] = {"original_key": k}
                elif v.get("type") in ["constant", "task"]:
                    data[group_name][part_name] = {"original_key": "task_index"}
                else:
                    raise NotImplementedError
        write_records(data, output_name)

    def _build_tasks(self, suffix: str = "jsonl"):
        assert self.src is not None, "Source Data Should be initialized"
        output_name = f"{self.output_dir}/meta/tasks.{suffix.lstrip('.')}"
        data: list[dict[str, Any]] = [{"task_index": 0, "task": ""}]
        for v in self.obs_config.values():
            if v.get("type") in ["constant", "task"]:
                data.append({"task_index": 1, "task": v.get("value")})
        assert len(data) == 2, "1 env 1 task"
        write_records(data, output_name)

    def _concat_state(self):
        for k, cfg in self.obs_config:
            pass
