"""RAPIDS-backed GPU paths for selected clustering stages.

Lazy-imported only when ``--device gpu`` is passed to ``bgc-cluster``. The
CLI calls :func:`ensure_gpu_available` before importing anything else from
this subpackage so we can fail fast with a clear error message.
"""

from __future__ import annotations


def ensure_gpu_available() -> dict:
    """Verify the GPU extras are installed and a CUDA device is visible.

    Returns a dict with diagnostic info (``cuda_version``, ``gpu_model``,
    ``library_versions``). Raises ``RuntimeError`` with a helpful message
    otherwise — no silent CPU fallback.
    """
    try:
        import cugraph  # noqa: F401
        import cuml  # noqa: F401
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GPU extras not installed. Re-install with "
            '`pip install "mgnify-bgcs-common-core[hpc-gpu]"` '
            "or run with --device cpu."
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "No CUDA runtime visible from this process. "
            "Check that this job is running on a GPU node and that "
            "nvidia-smi reports at least one device."
        ) from exc

    if device_count == 0:
        raise RuntimeError(
            "cupy.cuda.runtime.getDeviceCount() returned 0. "
            "Check that this job is running on a GPU node."
        )

    cuda_version = ""
    gpu_model = ""
    try:
        cuda_version = str(cp.cuda.runtime.runtimeGetVersion())
    except Exception:
        pass
    try:
        device = cp.cuda.Device(0)
        attrs = cp.cuda.runtime.getDeviceProperties(device.id)
        name = attrs.get("name") if isinstance(attrs, dict) else None
        if name is not None:
            gpu_model = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        pass

    return {
        "cuda_version": cuda_version,
        "gpu_model": gpu_model,
        "library_versions": _library_versions(),
    }


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for mod_name in ("cupy", "cuml", "cugraph"):
        try:
            mod = __import__(mod_name)
            out[mod_name] = getattr(mod, "__version__", "")
        except Exception:
            out[mod_name] = ""
    return out
