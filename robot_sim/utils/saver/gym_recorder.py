import copy
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
from loguru import logger

from .writers import write_records


class GymRecorder(gym.Wrapper):
    """Gym-style recorder that logs observations, actions, rewards, and info."""

    def __init__(
        self,
        env: gym.Env,
        record_reset: bool = True,
        copy_data: bool = True,
        include_render: bool = False,
        autosave: bool = False,
    ) -> None:
        super().__init__(env)
        self._record_reset = bool(record_reset)
        self._autosave = bool(autosave)
        self._include_render = bool(include_render)
        self._copy_data = bool(copy_data)

        self._records: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._episode_index = 0
        self._step_index = 0
        self._dirty = False

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._step_index = 0
        if self._record_reset:
            self._append_record(
                event="reset",
                observation=obs,
                action=None,
                reward=None,
                terminated=False,
                truncated=False,
                info=info,
            )
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._append_record(
            event="step",
            observation=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self._step_index += 1
        if terminated or truncated:
            self._episode_index += 1
            self._step_index = 0
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._dirty:
            self.save()
        super().close()

    def _append_record(
        self,
        *,
        event: str,
        observation: Any,
        action: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: Any,
    ) -> None:
        record = {
            "episode": int(self._episode_index),
            "step": int(self._step_index),
            "event": event,
            "timestamp": time.time(),
            "observation": self._clone(observation) if self._copy_data else observation,
            "action": action,
            "reward": self._clone(reward) if self._copy_data else reward,
            "terminated": self._clone(terminated) if self._copy_data else terminated,
            "truncated": self._clone(truncated) if self._copy_data else truncated,
            "info": self._clone(info) if self._copy_data else info,
        }
        if self._include_render:
            record["render"] = self._clone(self.env.render()) if self._copy_data else self.env.render()

        self._records[self._episode_index].append(record)
        self._dirty = True

        if self._autosave:
            self.save()

    def save(self, output_path: str | Path | None = None, format: str = "pkl") -> None:
        if not self._records:
            logger.warning("Recorder has no data to save.")
            return

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{tempfile.gettempdir()}/gym_recording_{ts}/recordings"

        output_format = Path(output_path).suffix.lstrip(".").lower()
        output_format = output_format if output_format else format
        output_name_prefix = str(Path(output_path).with_suffix(""))

        for k, v in self._records.items():
            logger.info(f"Saving episode {k} with {len(v)} steps.")
            write_records(path=f"{output_name_prefix}/episode_{k:06d}.{output_format}", records=v)
            if self._include_render:
                data = [rec["render"] for rec in v if "render" in rec]
                write_records(
                    path=Path(f"{output_name_prefix}/videos/episode_{k:06d}.mp4"),
                    records=data,
                    video_fps=self.env.metadata.get("render_fps", 30),
                )
        self._dirty = False
        logger.info(f"Recorder saved data to {output_path}")

    @staticmethod
    def _clone(value: Any) -> Any:
        try:
            return copy.deepcopy(value)
        except Exception:
            return value
