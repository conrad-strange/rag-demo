import os


def resolve_embedding_device(configured_device: str = "auto") -> str:
    requested = (os.getenv("EMBEDDING_DEVICE") or configured_device or "auto").strip().lower()
    if requested == "gpu":
        requested = "cuda"

    try:
        import torch
    except Exception:
        if requested not in ("auto", "cpu"):
            print(f"Embedding device {requested!r} requested, but torch is unavailable; using CPU.")
        return "cpu"

    cuda_available = torch.cuda.is_available()
    mps_available = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )

    if requested in ("", "auto"):
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if requested.startswith("cuda") and not cuda_available:
        print(f"Embedding device {requested!r} requested, but CUDA is unavailable; using CPU.")
        return "cpu"

    if requested == "mps" and not mps_available:
        print("Embedding device 'mps' requested, but MPS is unavailable; using CPU.")
        return "cpu"

    return requested
