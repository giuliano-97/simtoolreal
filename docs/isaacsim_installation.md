# IsaacSim (Isaac Lab) installation

The `isaacsimenvs/` package runs the SimToolReal task in Isaac Sim via Isaac Lab. It targets **Isaac Sim 6.1 / Isaac Lab 3.0 Early Access** (published as `isaaclab==3.0.0rc1`), which requires **Python 3.12**, and lives in a **second venv** at `.venv_isaacsim/`, separate from the Python 3.8 Isaac Gym venv that [isaacgym_installation.md](isaacgym_installation.md) sets up. The two environments are independent — never install Isaac Gym packages into `.venv_isaacsim` or vice versa.

Isaac Sim 5.x / Isaac Lab 2.3 is no longer supported: its RTX renderer crashes on current NVIDIA drivers (R6xx), so camera sensors and video recording only work on Isaac Sim 6.

## Prerequisites

- Python 3.12
- NVIDIA GPU with driver >= 570 (CUDA 12.8); tested with 610.43
- `uv` for package management ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))

## Install

There are two ways to set up `.venv_isaacsim/`, and both produce an equivalent environment: the original step-by-step `uv pip` install, or a single `uv sync` against a lockfile. Run either one from the repo root.

### Step-by-step (`uv pip`)

```bash
uv venv .venv_isaacsim --python 3.12

# PyTorch for CUDA 12.8 (Isaac Lab 3.0 is built against PyTorch 2.11)
uv pip install --python .venv_isaacsim/bin/python \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Vendored rl_games + inference deps
uv pip install --python .venv_isaacsim/bin/python -e ./rl_games/
uv pip install --python .venv_isaacsim/bin/python \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro "imageio[ffmpeg]" wandb termcolor

# Isaac Lab 3.0 EA + Isaac Sim 6.1 (~15 GB download; first launch builds RTX shaders, takes ~2-5 min).
# Isaac Sim's dependencies are split across pypi.nvidia.com and PyPI and include prereleases.
uv pip install --python .venv_isaacsim/bin/python \
  "isaaclab[isaacsim,video]==3.0.0rc1" --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow

# Offline collision decomposition (CoACD) + tyro CLI fix. Install AFTER isaaclab so it
# wins the resolution: tyro.cli (used by dextoolbench/generate_collision_meshes.py) needs
# NoExtraItems from typing_extensions>=4.13, but isaaclab pins 4.12.2.
uv pip install --python .venv_isaacsim/bin/python coacd "typing_extensions>=4.13"

# Register repo-local packages (isaacsimenvs, isaacgymenvs, deployment, ...)
uv pip install --python .venv_isaacsim/bin/python -e . --no-deps
```

`--no-deps` on the last line is required: the root `pyproject.toml` pins `numpy==1.23.0`, `warp-lang==0.10.1`, and `isaacgym-stubs`, which all conflict with Python 3.12 / Isaac Sim. The repo packages themselves install cleanly.

Keep the version pins exact (`isaaclab==3.0.0rc1`, torch 2.11.0+cu128): Isaac Lab 3.0 prereleases still change the asset, data and importer APIs that `isaacsimenvs` depends on. The `video` extra provides MoviePy, which Isaac Lab 3's video recording needs.

### One-step (`uv sync`)

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv_isaacsim" uv sync --project envs/isaacsim
```

This creates `.venv_isaacsim/` with Python 3.12 and installs the exact versions pinned in `envs/isaacsim/uv.lock` (~15 GB download; the first launch builds RTX shaders and takes ~2-5 min). Re-run the same command after pulling changes to bring the venv up to date. `UV_PROJECT_ENVIRONMENT` must be an absolute path, because uv resolves relative paths against `envs/isaacsim/`.

What `envs/isaacsim/pyproject.toml` sets up:

- PyTorch 2.11.0 for CUDA 12.8 from the PyTorch index.
- `isaaclab[isaacsim,video]==3.0.0rc1` (Isaac Lab 3.0 Early Access, which pins `isaacsim==6.1.0.0`) from `pypi.nvidia.com`. Keep the pin exact: Isaac Lab 3.0 prereleases still change the APIs that `isaacsimenvs` depends on.
- `index-strategy = "unsafe-best-match"` and `prerelease = "allow"`: Isaac Sim 6's dependencies are published partly on `pypi.nvidia.com` and partly on PyPI, and some are prereleases (e.g. `tinyobjloader==2.0.0rc13`).
- The vendored `rl_games/` (editable), plus the inference and CoACD dependencies.
- The repo-local packages (`isaacsimenvs`, `isaacgymenvs`, `deployment`, ...), installed editable. The env doesn't depend on the root `pyproject.toml`, which pins `numpy==1.23.0`, `warp-lang==0.10.1` and `isaacgym-stubs` for the Isaac Gym env; all of those conflict with Python 3.12 / Isaac Sim. If you add a top-level package to the root `pyproject.toml`, add it to `envs/isaacsim/pyproject.toml` too.
- An override that raises `typing_extensions` to `>=4.13`. `isaaclab` pins 4.12.2, but `tyro.cli` (used by `dextoolbench/generate_collision_meshes.py`) needs `NoExtraItems`, which first appeared in 4.13.

To change dependencies, edit `envs/isaacsim/pyproject.toml`, run `uv lock --project envs/isaacsim`, and commit the updated `uv.lock`.

### Verify

This imports `isaaclab`, which asks for the EULA unless it's accepted through the env var. It only checks that `isaacsimenvs` can be found and doesn't import it, because `isaacsimenvs` needs `AppLauncher` to run first (see Gotchas):

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -c "
import importlib.util, torch, isaaclab
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('isaaclab:', isaaclab.__file__)
print('isaacsimenvs:', importlib.util.find_spec('isaacsimenvs').origin)
"
```

## Running

Non-interactive launches need the EULA env var:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Optional speedup — point the Omniverse shader cache at a local SSD instead of NFS:

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"
```

Smoke tests (one Kit boot per process, ~1-2 min each; run individually, not via pytest):

```bash
.venv_isaacsim/bin/python isaacsimenvs/tests/test_load_isaacsim.py
.venv_isaacsim/bin/python isaacsimenvs/tests/test_gym_register.py
.venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \
  --num_envs 8 --num_assets_per_type 2 --steps 10
.venv_isaacsim/bin/python isaacsimenvs/tests/test_empty_scene.py --video   # camera rendering
```

Training:

```bash
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_cfg_entry_point
```

Isaac Lab 3 runs headless unless a visualizer is selected, and its launcher no longer has `--headless` or `--enable_cameras`:

- `--viz kit` opens the Kit viewer (`--viz none` disables all visualizers).
- `train.py --enable_cameras` renders camera sensors (student cameras).
- `train.py --capture_video` records mp4 clips from a headless Kit viewport into the run directory (`--video_capture_frames` env steps per clip, a new clip every `--video_interval` env steps).

## Gotchas

- **AppLauncher before isaaclab imports.** Isaac Lab sub-namespaces (`isaaclab.sim`, `isaaclab.envs`, ...) only resolve after `AppLauncher(args)` runs. Any script importing `isaaclab.*` must instantiate `AppLauncher` first.
- **Kit shutdown can hang** after the work is done. Scripts flush stdout/stderr and call `os._exit(0)` rather than waiting for a clean Kit teardown.
- **First SimToolReal startup is slow**: object URDFs are procedurally generated and converted to USD per run (no cache); startup scales with `num_assets_per_type`.
- **One Isaac Sim instance per GPU.** Booting a second Kit process while another is running on the same GPU can crash the booting one mid-startup. Wait for training/eval/demo processes to finish before launching another.
- **Isaac Lab 3 conventions.** Quaternions are `(x, y, z, w)` everywhere (Isaac Lab 2 used `(w, x, y, z)`); this is also the Isaac Gym / policy order, and all `*_quat_xyzw` config fields, `reset.fixed_start_pose` / `fixed_goal_pose` (`x, y, z, qx, qy, qz, qw`) and `quat_xyzw` trajectory files follow it. Asset data is a `ProxyArray`: read it with `.torch` (e.g. `robot.data.joint_pos.torch`). See the [Isaac Lab 3.0 migration guide](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/migration/migrating_to_isaaclab_3-0.html).
- **URDF importer 3.0 workarounds** live in `isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py`: absolute URDF paths (relative ones silently lose their meshes), a fix for the importer's fixed-joint merge at pitch = -90° (it mirrored the Sharpa fingertip pads), cylinder colliders turned back into capsules, a fixed-base articulation root, and joint velocity limits copied to PhysX.
