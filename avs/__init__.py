__all__ = ["__version__"]

__version__ = "0.1.0"

# Best-effort: make repo-local ffmpeg available for all subprocess calls.
# This avoids requiring sudo/apt-get in constrained environments.
try:
    from avs.utils.ffmpeg import ensure_ffmpeg_in_path

    ensure_ffmpeg_in_path()
except Exception:
    # Keep imports robust; ffmpeg is only needed for some pipelines.
    pass
