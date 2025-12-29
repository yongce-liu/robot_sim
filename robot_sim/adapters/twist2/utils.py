import numpy as np
import torch


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))

    @classmethod
    def load_onnx_policy(cls, policy_path: str, device: str) -> "OnnxPolicyWrapper":
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
        providers = []
        available = ort.get_available_providers()
        if device.startswith("cuda"):
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            else:
                print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
        providers.append("CPUExecutionProvider")
        session = ort.InferenceSession(policy_path, providers=providers)
        input_name = session.get_inputs()[0].name
        print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
        return OnnxPolicyWrapper(session, input_name)
