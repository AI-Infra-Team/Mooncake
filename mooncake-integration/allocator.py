import os
import threading
from importlib import resources
from typing import Dict, Final, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import device as torch_device
    from torch.cuda.memory import CUDAPluggableAllocator
else:
    torch_device = Any  # type: ignore
    CUDAPluggableAllocator = Any  # type: ignore


class NVLinkAllocator:
    _instances: Dict[torch_device, CUDAPluggableAllocator] = {}
    _lock: Final = threading.Lock()

    @classmethod
    def _get_so_path(cls) -> str:
        """Dynamically locate nvlink_allocator.so in the mooncake package installation"""
        try:
            # Attempt to locate package resource
            with resources.path("mooncake", "nvlink_allocator.so") as so_path:
                if so_path.exists():
                    return str(so_path)
        except (ImportError, FileNotFoundError, TypeError):
            pass

        # Fallback strategy: check in package location via import metadata
        try:
            import mooncake

            base_path = os.path.dirname(os.path.abspath(mooncake.__file__))
            so_path = os.path.join(base_path, "nvlink_allocator.so")
            if os.path.exists(so_path):
                return so_path
        except (ImportError, FileNotFoundError, TypeError):
            raise ImportError(
                "SGLANG_MOONCAKE_CUSTOM_MEM_POOL require mooncake-transfer-engine >= 0.3.3.post2."
            )

    @classmethod
    def get_allocator(cls, device: torch_device) -> CUDAPluggableAllocator:
        # Import torch lazily to avoid hard dependency unless used
        try:
            from torch.cuda.memory import CUDAPluggableAllocator as _CUDAPluggableAllocator  # type: ignore
        except Exception as e:
            import logging
            msg = (
                "[Mooncake] PyTorch not installed; NVLinkAllocator requires PyTorch.\n"
                "Install PyTorch to enable custom CUDA memory pool: pip install torch"
            )
            logging.warning(msg)
            print(msg)
            raise ImportError("PyTorch is required to use NVLinkAllocator") from e

        with cls._lock:
            if device not in cls._instances:
                so_path = cls._get_so_path()
                cls._instances[device] = _CUDAPluggableAllocator(
                    so_path, "mc_nvlink_malloc", "mc_nvlink_free"
                )
            return cls._instances[device]
