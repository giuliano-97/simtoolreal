# Agent guide

Notes for coding agents working in this repo (`CLAUDE.md` is a symlink to this file). The README covers the project itself. This file covers how to set up the Isaac Sim environment, how to run things, and the pitfalls that are easy to hit.

## Branch

This fork's work lives on `dev/insait` (remote `fork`, https://github.com/giuliano-97/simtoolreal). It adds two things on top of upstream `tylerlum/simtoolreal`:

- **Isaac Lab 3 port:** the Isaac Sim backend (`isaacsimenvs/`) runs on Isaac Sim 6.1 / Isaac Lab 3.0 Early Access (`isaaclab==3.0.0rc1`). Isaac Sim 5.x is not supported, because its RTX renderer crashes on current drivers.
- **Flexiv Rizon 4 + left Sharpa hand** as a second robot next to the KUKA iiwa14. See `docs/flexiv_rizon4.md`.

## Setup

Requirements:
- Linux with an NVIDIA driver >= 570;
- [`uv`](https://docs.astral.sh/uv/);
- about 30 GB of disk for the venv and the uv cache.

From the repo root:

```bash
git checkout dev/insait
UV_PROJECT_ENVIRONMENT="$PWD/.venv_isaacsim" uv sync --project envs/isaacsim
```

- **Where it installs:** this creates `.venv_isaacsim/` (Python 3.12) from `envs/isaacsim/uv.lock`, which pins Isaac Sim 6.1, Isaac Lab 3.0.0rc1 and torch 2.11+cu128. The download is about 15 GB.
- **Re-run** the same command after pulling.
- **`UV_PROJECT_ENVIRONMENT` must be absolute.**
- **Changing dependencies:** edit `envs/isaacsim/pyproject.toml`, then run `uv lock --project envs/isaacsim`. Don't install the root `pyproject.toml` into this venv; it is for the legacy Python 3.8 Isaac Gym env.

The full guide and the step-by-step alternative are in `docs/isaacsim_installation.md`.

Optional downloads (not in git):

```bash
.venv_isaacsim/bin/python download_pretrained_policy.py      # -> pretrained_policy/ (KUKA checkpoint)
.venv_isaacsim/bin/python download_dextoolbench_data.py
```

### Verify

The import check works without a GPU:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -c "
import importlib.util, torch, isaaclab
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('isaacsimenvs', importlib.util.find_spec('isaacsimenvs').origin)"
```

The following need a GPU:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
.venv_isaacsim/bin/python isaacsimenvs/tests/test_load_isaacsim.py
.venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py --num_envs 8 --num_assets_per_type 2 --steps 10
.venv_isaacsim/bin/python isaacsimenvs/tests/test_robot_spec_validation.py --robot flexiv_rizon4
```

## Running

- **Accept the EULA:** always `export OMNI_KIT_ACCEPT_EULA=YES`. Without it, Kit stops at an interactive license prompt.
- **One Kit process per GPU.** Booting a second Isaac Sim process on a busy GPU can crash it.
- **Startup time:** the first launch builds RTX shaders (2–5 min). To keep the shader cache on a fast local disk, set `OMNI_KIT_CACHE_PATH`.
- **Test scripts are standalone Kit scripts.** Run them one per process, not through pytest.
  - They end with `os._exit(0)` because Kit teardown can hang, so a crash can still exit 0.
  - Judge them by their final `... OK` line.
- **Training:**
  - Pick the robot with `env.assets.robot=kuka_iiwa14|flexiv_rizon4`. The default is the KUKA.
  - The paper recipe is SAPG with `env.scene.num_envs=24576` and `agent.params.config.expl_coef_block_size=4096`. If you use fewer envs, keep `num_envs / expl_coef_block_size = 6`.
  - Example:
    ```bash
    .venv_isaacsim/bin/python isaacsimenvs/train.py \
      --task Isaacsimenvs-SimToolReal-Direct-v0 --agent rl_games_sapg_cfg_entry_point \
      env.assets.robot=flexiv_rizon4 env.scene.num_envs=24576 \
      agent.params.config.expl_coef_block_size=4096
    ```
- **Flags:** runs are headless by default.
  - `--viz kit` opens a window.
  - `train.py --enable_cameras` turns on camera sensors.
  - `--capture_video` records clips.
  - Logging to W&B needs `--wandb_activate` and `wandb login`.
- **Clean-up:** each run procedurally generates object assets into `/tmp/simtoolreal_assets_*` and doesn't delete them.

## Clusters

- **Login nodes may have no GPU.** Install there, but run anything that boots Isaac Sim inside a GPU job (`srun`/`sbatch`).
- **RTX cameras need a GPU with RT cores** (e.g. RTX 6000, L40, RTX 3090/4090). A100/H100/H200 have none. That covers student cameras, `--capture_video` and `--render`. Training without cameras doesn't render, but it hasn't been tested on those GPUs here.
- **Home and uv cache:** if home is small or on slow network storage, point `UV_CACHE_DIR` and `OMNI_KIT_CACHE_PATH` at scratch.

## Isaac Lab 3 pitfalls

- **Import order:** `AppLauncher(args)` must run before any `isaaclab.*` sub-module import, and `isaacsimenvs` imports them. Every entry point in this repo follows that order.
- **Quaternions** are `(x, y, z, w)` everywhere, including config fields named `*_quat_xyzw` and trajectory files.
- **Asset data** is a `ProxyArray`. Read it with `.torch`, e.g. `robot.data.joint_pos.torch`.
- **Cameras:**
  - A `Camera` needs one prim per env (`/World/envs/env_[^/]+/Cam`). A single `/World/Cam` errors when `num_envs > 1`.
  - Per-env RTX scene partitioning is on by default, so an env's camera sees only its own env. A camera outside the envs sees only the ground.
  - For multi-env overview renders, set `ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0`. Training and the student camera keep the default.
- **URDF importer 3.0 workarounds** live in `isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py`. Keep them when touching robot loading:
  - absolute URDF paths, because relative ones silently lose their meshes;
  - a fix for mirrored fingertip pads after the fixed-joint merge;
  - capsule colliders;
  - a fixed-base articulation root;
  - joint velocity limits copied to PhysX.
- **Version pins:** keep `isaaclab==3.0.0rc1` exact. Isaac Lab 3 prereleases still change the APIs used here. Port by following the official release notes and migration guide.

## Robots

- **Robot specs:** `simtoolreal_shared/robots.py` is the single source of the arm-specific values: URDF, joint and link names, default and raised poses, PD gains, base pose and palm-center offset.
- **Rizon URDF:** `assets/urdf/rizon4_sharpa_description/rizon4_left_sharpa.urdf` is generated by `scripts/generate_rizon4_sharpa_urdf.py`. Don't edit it by hand.
  - After changing the mount, regenerate it.
  - Then run `scripts/compute_rizon4_sharpa_defaults.py` and paste the values it prints into `FLEXIV_RIZON4`.
  - Validate with `test_robot_spec_validation.py --robot flexiv_rizon4`.
- **Still placeholders on the Rizon,** all listed in `docs/flexiv_rizon4.md`:
  - the hand yaw on the flange;
  - the arm PD gains (KUKA values);
  - the base pose relative to the table.
- **Pretrained checkpoint:** `pretrained_policy/` is KUKA-only. A Rizon policy has to be trained.
- **Deployment:** `deployment/` and the legacy Isaac Gym env (`isaacgymenvs/`) still assume the KUKA.
