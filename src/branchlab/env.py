import os

import torch

import branchlab  # noqa: F401  (sets LIBERO_CONFIG_PATH / MUJOCO_GL before libero loads)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def get_task(suite: str, task_id: int):
    return benchmark.get_benchmark_dict()[suite]().get_task(task_id)


def load_init_states(suite: str, task_id: int):
    task = get_task(suite, task_id)
    path = os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )
    # LIBERO's own loader calls torch.load without weights_only=False, which
    # torch>=2.6 rejects for these numpy-array pickles.
    return torch.load(path, weights_only=False)


def make_env(suite: str, task_id: int, image_size: int = 256, **kwargs):
    task = get_task(suite, task_id)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=image_size,
        camera_widths=image_size,
        **kwargs,
    )
    return env, task
