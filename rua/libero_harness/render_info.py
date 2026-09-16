"""Create a tiny offscreen context; no simulator, model, or robot actions."""
import json
import os

import mujoco
from OpenGL import GL


def main():
    context = mujoco.GLContext(64, 64)
    try:
        context.make_current()
        print(json.dumps({
            "backend": os.environ.get("MUJOCO_GL"),
            "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
            "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
            "version": GL.glGetString(GL.GL_VERSION).decode(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }), flush=True)
    finally:
        context.free()


if __name__ == "__main__":
    main()
