#!/usr/bin/env python3
"""Script to run simulations with threaded visualization."""

import queue
import threading
from pathlib import Path

import cv2
import hydra
import numpy as np
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import robot_sim
from robot_sim.backends import BackendFactory
from robot_sim.configs import SimulatorConfig
from robot_sim.utils.helper import setup_logger

PROJECT_DIR = Path(robot_sim.__file__).parents[1].resolve()


class DualImageViewer:
    """Thread-safe dual image viewer using OpenCV - displays two images side by side."""

    def __init__(self, window_name: str = "Robot Cameras", max_queue_size: int = 2):
        self.window_name = window_name
        self.image_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        """Start the viewer thread."""
        self.thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self.thread.start()
        logger.info(f"Dual image viewer thread started. Window: {self.window_name}")

    def _viewer_loop(self):
        """Main loop for displaying images in separate thread."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set():
            try:
                # Get image pair from queue with timeout
                image1, image2 = self.image_queue.get(timeout=0.1)

                # Process both images
                processed_images = []
                for image in [image1, image2]:
                    if isinstance(image, np.ndarray):
                        # Convert to uint8 if needed
                        if image.dtype != np.uint8:
                            if image.max() <= 1.0:
                                image = (image * 255).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)

                        # Convert RGB to BGR
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        else:
                            image_bgr = image

                        processed_images.append(image_bgr)

                # Combine images horizontally
                if len(processed_images) == 2:
                    # Add text labels
                    img1 = processed_images[0].copy()
                    img2 = processed_images[1].copy()

                    # Add labels to images
                    cv2.putText(img1, "G1 Head Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img2, "G2 Head Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    combined = np.hstack([img1, img2])
                    cv2.imshow(self.window_name, combined)

            except queue.Empty:
                pass

            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("'q' key pressed in viewer window")
                self.stop_event.set()
                break

        cv2.destroyAllWindows()
        logger.info("Dual image viewer thread stopped")

    def update_images(self, image1: np.ndarray, image2: np.ndarray):
        """Add new image pair to display queue (non-blocking)."""
        try:
            # If queue is full, remove old images and add new ones
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
            self.image_queue.put_nowait((image1, image2))
        except queue.Full:
            pass  # Skip frame if queue is full

    def stop(self):
        """Stop the viewer thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def is_stopped(self) -> bool:
        """Check if viewer has been stopped."""
        return self.stop_event.is_set()


@hydra.main(version_base=None, config_path=str(PROJECT_DIR), config_name="/configs/simulator.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point with threaded visualization.

    Switch backends easily:
        python scripts/run_sim_threaded.py backend=isaac
        python scripts/run_sim_threaded.py backend=mujoco

    Args:
        cfg: Hydra configuration object
    """
    setup_logger(max_file_size=10)
    # Print configuration
    cfg = SimulatorConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
    cfg_dict = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    logger.info("Configuration:\n{}", yaml.dump(cfg_dict))

    # Create simulation manager
    sim_backend = BackendFactory(config=cfg).backend

    # Setup and run simulation
    sim_backend.launch()
    logger.info("Simulation launched successfully.")

    default_state = sim_backend.get_states()

    # Start a single dual image viewer
    viewer = DualImageViewer(window_name="Robot Cameras (G1 | G2)", max_queue_size=2)
    viewer.start()

    try:
        while not viewer.is_stopped():
            sim_backend.set_states(default_state)
            sim_backend.simulate()
            states = sim_backend.get_states()
            try:
                image1 = states["g1"].sensors["head_camera"]["rgb"]
            except KeyError:
                image1 = np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                image2 = states["g2"].sensors["head_camera"]["rgb"]
            except KeyError:
                image2 = np.zeros((480, 640, 3), dtype=np.uint8)
            # Send both images to the dual viewer
            viewer.update_images(image1, image2)

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    finally:
        viewer.stop()
        sim_backend.close()
        logger.info("Simulation closed.")


if __name__ == "__main__":
    main()
