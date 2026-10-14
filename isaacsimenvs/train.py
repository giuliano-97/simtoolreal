"""Train an isaacsimenvs task with our vendored rl_games.

Pipeline:
    argparse (--task, --agent, AppLauncher, wandb/video flags)
        ↓
    AppLauncher (boots Kit; must precede any isaaclab.* import — see CLAUDE.md)
        ↓
    @hydra_task_config_with_yaml  (configclass defaults ← task YAML overlay ← Hydra CLI)
        ↓
    gym.make(task_id, cfg=env_cfg)  → DirectRLEnv wrapped by gym.Wrapper
        ↓
    isaaclab_rl.RlGamesVecEnvWrapper (via register_rlgames_env — clipping,
                                      device bridging, obs-group routing)
        ↓
    rl_games.torch_runner.Runner (PPO / SAPG — both live in ./rl_games/)

CLI shape:
    python isaacsimenvs/train.py \
        --task Isaacsimenvs-SimToolReal-Direct-v0 \
        --agent rl_games_sapg_cfg_entry_point \   # or rl_games_cfg_entry_point
        --capture_viewer \
        --wandb_activate --wandb_project X --wandb_name Y \
        env.scene.num_envs=4096 \
        agent.params.config.max_epochs=200 \
        agent.params.config.minibatch_size=16384 \
        agent.params.seed=42
"""

from __future__ import annotations

import argparse
import math
import os
import sys


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train isaacsimenvs task via rl_games.")
    # --- Task/agent selection ---
    parser.add_argument("--task", required=True, help="Gym task id, e.g. Isaacsimenvs-SimToolReal-Direct-v0")
    parser.add_argument(
        "--agent",
        default="rl_games_cfg_entry_point",
        help="Key in gym.register kwargs for the rl_games YAML. "
        "Use rl_games_sapg_cfg_entry_point for SAPG.",
    )
    # --- Runtime toggles ---
    parser.add_argument("--test", action="store_true", help="Run inference (player) instead of training")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth to restore")
    parser.add_argument(
        "--checkpoint_load_mode",
        choices=("resume", "weights"),
        default="resume",
        help="resume restores optimizer/rollout/env state; weights starts fresh from model weights.",
    )
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    # --- Video ---
    parser.add_argument(
        "--capture_video",
        action="store_true",
        help="Record mp4 clips from a headless Kit viewport; implies --enable_cameras",
    )
    parser.add_argument(
        "--video_interval", type=int, default=1000, help="Env steps between clip starts (>= --video_capture_frames)"
    )
    parser.add_argument("--video_capture_frames", type=int, default=120, help="Env steps per clip")
    parser.add_argument("--video_fps", type=int, default=30)
    # --- Pose-only interactive HTML viewer (no cameras / no renderer) ---
    parser.add_argument(
        "--capture_viewer",
        action="store_true",
        help="Write periodic pose-only interactive HTML viewers; does not enable Isaac cameras.",
    )
    parser.add_argument("--capture_viewer_len", type=int, default=600)
    parser.add_argument("--capture_viewer_interval", type=int, default=6000)
    parser.add_argument("--capture_viewer_env_id", type=int, default=0)
    parser.add_argument("--capture_viewer_wandb_key", default="interactive_viewer")
    parser.add_argument(
        "--capture_viewer_github_raw_base",
        default="",
        help="GitHub raw base URL used by the browser to fetch robot URDF meshes.",
    )
    parser.add_argument(
        "--capture_viewer_url_check",
        choices=("skip", "warn", "error"),
        default="skip",
        help="Whether to HEAD-check the robot URDF URL before writing viewer HTML.",
    )
    # --- wandb ---
    parser.add_argument("--wandb_activate", action="store_true")
    parser.add_argument("--wandb_project", default="isaacsimenvs")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="", help="Defaults to agent_cfg.params.config.name")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_notes", default="")
    parser.add_argument("--wandb_logcode_dir", default="")
    # --- AppLauncher flags (--viz, --device, --kit_args, etc.) ---
    AppLauncher.add_app_launcher_args(parser)
    # Isaac Lab 3 runs headless unless a visualizer is selected (--viz kit), and
    # its AppLauncher dropped the --enable_cameras flag while still reading the
    # option, which camera sensors need. It must be added after the AppLauncher
    # args, which reject parsers that already define their options.
    parser.add_argument(
        "--enable_cameras", action="store_true", help="Render camera sensors (student cameras, video)."
    )
    args_cli, hydra_args = parser.parse_known_args()

    # Recording a video requires cameras even if user forgot --enable_cameras.
    if args_cli.capture_video:
        args_cli.enable_cameras = True
        # Isaac Lab's recorder starts a new clip on every interval, cutting the
        # current one short.
        if args_cli.video_interval < args_cli.video_capture_frames:
            parser.error("--video_interval must be >= --video_capture_frames")

    # Hand the leftover key=value tokens to Hydra via sys.argv.
    sys.argv = [sys.argv[0]] + hydra_args

    app = AppLauncher(args_cli).app

    # 2. Safe to import isaaclab-backed modules now.
    import gymnasium as gym
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    from rl_games.torch_runner import Runner

    import isaacsimenvs  # noqa: F401  triggers gym.register side effects
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.utils.rlgames_utils import (
        EnvStatsAlgoObserver,
        MultiObserver,
        register_rlgames_env,
    )

    @hydra_task_config_with_yaml(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg: dict) -> None:
        hydra_run_dir = HydraConfig.get().runtime.output_dir

        # sim_device CLI flag still wins — it's a launcher-level concern, not
        # something we expect in the task YAML.
        env_cfg.sim.device = args_cli.sim_device
        # Keep simulator/object randomization and the policy runner on the same
        # seed. Previously only rl_games saw the agent seed override.
        env_cfg.seed = int(agent_cfg["params"]["seed"])

        if args_cli.capture_video:
            # Isaac Lab 3 records inside the env from a visualizer: one headless
            # Kit viewport (a single render product, num_envs-independent) framed
            # by env_cfg.sim.default_visualizer_cfg. A Camera sensor added after
            # env init would need another sim.reset(), which momentarily doubles
            # the PhysX scene state (~400 GB at 24576 envs).
            from pathlib import Path

            from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
            from isaaclab_visualizers.kit import KitVisualizerCfg

            env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, window_width=640, window_height=480)]
            env_cfg.video_recorders = [
                VideoRecorderCfg(
                    source="visualizer:kit",
                    output_dir=str(Path(hydra_run_dir) / "videos"),
                    fps=args_cli.video_fps,
                    video_length=args_cli.video_capture_frames,
                    video_interval=args_cli.video_interval,
                )
            ]

        env = gym.make(args_cli.task, cfg=env_cfg)

        if args_cli.capture_viewer:
            from pathlib import Path

            from isaacsimenvs.tasks.simtoolreal.pose_viewer import SimToolRealPoseViewerWrapper

            env = SimToolRealPoseViewerWrapper(
                env,
                output_dir=Path(hydra_run_dir) / "interactive_viewer",
                capture_len=args_cli.capture_viewer_len,
                capture_interval=args_cli.capture_viewer_interval,
                env_id=args_cli.capture_viewer_env_id,
                wandb_key=args_cli.capture_viewer_wandb_key,
                github_raw_base=args_cli.capture_viewer_github_raw_base,
                url_check=args_cli.capture_viewer_url_check,
            )

        # Clip bounds live in the rl_games YAML (params.env.*). Default to
        # +inf if absent so a task without clip YAML just runs unbounded —
        # matches isaacgymenvs's `cfg["env"].get("clipObservations", np.Inf)`.
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
        register_rlgames_env(
            env,
            rl_device=args_cli.rl_device,
            clip_obs=clip_obs,
            clip_actions=clip_actions,
        )

        observers = [EnvStatsAlgoObserver()]
        if args_cli.wandb_activate:
            from isaacsimenvs.utils.wandb_utils import WandbAlgoObserver

            # WandbAlgoObserver expects attribute access (cfg.wandb_project,
            # cfg.wandb_notes, …) and also passes cfg to omegaconf_to_dict
            # for wandb.config upload. OmegaConf satisfies both.
            wandb_cfg = OmegaConf.create(
                {
                    "wandb_activate": True,
                    "wandb_project": args_cli.wandb_project,
                    "wandb_group": args_cli.wandb_group,
                    "wandb_entity": args_cli.wandb_entity,
                    "wandb_name": args_cli.wandb_name or agent_cfg["params"]["config"]["name"],
                    "wandb_tags": list(args_cli.wandb_tags),
                    "wandb_notes": args_cli.wandb_notes,
                    "wandb_logcode_dir": args_cli.wandb_logcode_dir,
                }
            )
            observers.append(WandbAlgoObserver(wandb_cfg))

        runner = Runner(MultiObserver(observers))
        # Co-locate rl_games artifacts (checkpoints, summaries) with the Hydra
        # run dir so slurm logs + config + videos all live together.
        agent_cfg["params"]["config"]["train_dir"] = hydra_run_dir
        agent_cfg["params"]["config"]["device"] = args_cli.rl_device
        agent_cfg["params"]["config"]["device_name"] = args_cli.rl_device

        runner.load(agent_cfg)
        runner.reset()
        runner.run(
            {
                "train": not args_cli.test,
                "play": args_cli.test,
                "checkpoint": args_cli.checkpoint,
                "checkpoint_load_mode": args_cli.checkpoint_load_mode,
            }
        )

    run()

    # Kit shutdown hangs (per CLAUDE.md + isaacsim_conversion/distill.py).
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
