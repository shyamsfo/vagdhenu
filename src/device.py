"""Backend selection for Vāgdhenu inference — CUDA, Apple Metal (MPS), or CPU.

Import this BEFORE torch. PyTorch reads PYTORCH_ENABLE_MPS_FALLBACK once, when the MPS
dispatch table is registered at `import torch`, so setting it afterwards has no effect.
The fallback matters because the mel front-end and the vocos ISTFT head both call FFT
kernels whose Metal coverage varies by torch build; without it a missing kernel raises
instead of quietly running on CPU.

Override with VAGDHENU_DEVICE=cpu|mps|cuda|cuda:1 to pin a backend by hand.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(prefer=None):
    """Return the torch device string to run on. `prefer` wins unless it is unavailable."""
    import torch

    requested = prefer or os.environ.get("VAGDHENU_DEVICE", "").strip()
    if requested and _available(requested):
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _available(name):
    import torch

    if name.startswith("cuda"):
        return torch.cuda.is_available()
    if name.startswith("mps"):
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return name.startswith("cpu")


def describe(device):
    """One-line human summary of the chosen backend, for boot logs."""
    import torch

    if device.startswith("cuda"):
        return f"{device} ({torch.cuda.get_device_name(device if ':' in device else 0)})"
    if device.startswith("mps"):
        return "mps (Apple Metal — expect slower-than-realtime synthesis)"
    return "cpu (very slow; minutes per hemistich)"
