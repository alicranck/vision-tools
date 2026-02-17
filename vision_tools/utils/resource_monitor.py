"""
Resource monitoring utilities for detecting available hardware 
and recommending optimal scaling configurations.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ComputeBackend(Enum):
    """Available compute backends for model inference."""
    CUDA = "cuda"           # NVIDIA GPU with CUDA
    OPENVINO = "openvino"   # Intel CPU optimization
    ONNX = "onnx"           # Cross-platform fallback
    CPU = "cpu"             # Basic PyTorch CPU


@dataclass
class SystemResources:
    """Detected system resources and scaling recommendations."""
    cpu_cores: int
    ram_gb: float
    has_gpu: bool
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    recommended_backend: ComputeBackend = ComputeBackend.CPU
    recommended_batch_size: int = 1
    recommended_workers: int = 1


def get_system_resources() -> SystemResources:
    """
    Detect available system resources and return scaling recommendations.
    Designed to work reliably in serverless/cloud environments.
    
    Returns:
        SystemResources with detected hardware and recommendations.
    """
    # Detect CPU and RAM
    cpu_cores, ram_gb = _detect_cpu_ram()
    
    # Detect GPU
    has_gpu, gpu_name, gpu_vram_gb = _detect_gpu()
    
    # Determine recommended backend
    if has_gpu:
        backend = ComputeBackend.CUDA
        batch_size = _calculate_batch_size(gpu_vram_gb)
        workers = 1  # GPU handles parallelism internally
    else:
        backend = _get_cpu_backend()
        batch_size = 1
        workers = _calculate_recommended_workers(cpu_cores, ram_gb)
    
    resources = SystemResources(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        recommended_backend=backend,
        recommended_batch_size=batch_size,
        recommended_workers=workers,
    )
    
    logger.info(f"Detected resources: {resources}")
    return resources


def _detect_cpu_ram() -> tuple[int, float]:
    """Detect CPU cores and RAM, with fallbacks for restricted environments."""
    cpu_cores = 1
    ram_gb = 4.0  # Conservative default
    
    try:
        import psutil
        cpu_cores = psutil.cpu_count(logical=True) or 1
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        logger.warning("psutil not available, using defaults for CPU/RAM")
    except Exception as e:
        logger.warning(f"Failed to detect CPU/RAM: {e}")
    
    return cpu_cores, ram_gb


def _detect_gpu() -> tuple[bool, Optional[str], Optional[float]]:
    """Detect NVIDIA GPU availability and VRAM."""
    has_gpu = False
    gpu_name = None
    gpu_vram_gb = None
    
    # Try torch.cuda first (most reliable for our use case)
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
            # Get VRAM via torch
            props = torch.cuda.get_device_properties(0)
            gpu_vram_gb = props.total_memory / (1024 ** 3)
            return has_gpu, gpu_name, gpu_vram_gb
    except Exception as e:
        logger.debug(f"torch.cuda detection failed: {e}")
    
    # Fallback to pynvml for more accurate VRAM (optional dependency)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_vram_gb = mem_info.total / (1024 ** 3)
        has_gpu = True
        pynvml.nvmlShutdown()
    except ImportError:
        logger.debug("pynvml not available")
    except Exception as e:
        logger.debug(f"pynvml detection failed: {e}")
    
    return has_gpu, gpu_name, gpu_vram_gb


def _get_cpu_backend() -> ComputeBackend:
    """Determine optimal CPU backend based on processor type."""
    try:
        # Check if running on Intel CPU -> use OpenVINO
        import platform
        proc = platform.processor().lower()
        if "intel" in proc or "genuine" in proc:
            return ComputeBackend.OPENVINO
    except Exception:
        pass
    
    # Try to detect Intel via cpuinfo if available
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        vendor = info.get("vendor_id_raw", "").lower()
        if "intel" in vendor or "genuineintel" in vendor:
            return ComputeBackend.OPENVINO
    except ImportError:
        pass
    except Exception:
        pass
    
    # Default to OpenVINO - it works on AMD too, just optimized for Intel
    return ComputeBackend.OPENVINO


def _calculate_batch_size(gpu_vram_gb: Optional[float]) -> int:
    """Calculate recommended batch size based on GPU VRAM."""
    if gpu_vram_gb is None:
        return 1
    
    # Conservative estimates for vision models
    if gpu_vram_gb >= 24:
        return 16
    elif gpu_vram_gb >= 16:
        return 8
    elif gpu_vram_gb >= 8:
        return 4
    elif gpu_vram_gb >= 4:
        return 2
    else:
        return 1


def _calculate_recommended_workers(cpu_cores: int, ram_gb: float) -> int:
    """Recommend workers based on CPU cores, leaving headroom for system."""
    # Use ~75% of cores, minimum 1, leave 1 core for system
    workers = max(1, int(cpu_cores * 0.75) - 1)
    # Cap based on RAM (assume ~2GB per worker for vision models)
    ram_cap = max(1, int(ram_gb / 2))
    return min(workers, ram_cap)
