import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_ROOT = PROJECT_ROOT / "third_party" / "LIBERO" / "libero" / "libero"

# Must be set before `libero` / `robosuite` are imported. LIBERO otherwise
# prompts on stdin and writes its config to ~/.libero.
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
