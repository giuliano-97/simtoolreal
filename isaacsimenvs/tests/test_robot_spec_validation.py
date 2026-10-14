"""Validate a robot spec (simtoolreal_shared/robots.py) in Isaac Sim.

Checks, with reset noise and DR disabled:

  1. The palm center the policy observes at reset matches forward kinematics
     of the URDF (base pose + default joints + palm offset), i.e. URDF->USD
     conversion, fixed-joint merging, base placement and palm offset agree.
  2. Holding the default and raised poses for 2 s keeps every joint within
     HOLD_TOL_RAD of its target (no self-collision pushing the robot around).
  3. Arm PD step response, one joint at a time (+0.1 rad): reports overshoot,
     rise time and steady-state error; fails if a joint does not settle.
  4. A random-action rollout stays finite, with bounded joint velocities.
  5. With --render: saves PNGs of the default pose from two viewpoints.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_robot_spec_validation.py \\
      --robot flexiv_rizon4 [--render --out_dir /tmp/robot_validation]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLD_TOL_RAD = 0.01
STEP_RAD = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="kuka_iiwa14", help="simtoolreal_shared.robots key")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--random_steps", type=int, default=300)
    parser.add_argument("--render", action="store_true", help="Save PNGs of the default pose")
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "outputs/robot_validation"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.render:
        args.enable_cameras = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import yourdfpy
    from scipy.spatial.transform import Rotation

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import OBS_FIELD_SIZES

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = 1
    cfg.assets.robot = args.robot
    rs = cfg.reset
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = 0.0
    # No timeouts mid-measurement.
    cfg.episode_length_s = 1.0e4
    cfg.termination.episode_length = 10**9

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    if args.render:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg

        # One camera per env: Lab 3 RTX scene partitioning hides env content from cameras outside the envs.
        camera = Camera(CameraCfg(
            prim_path="/World/envs/env_[^/]+/ValidationCamera", update_period=0, height=768, width=1024, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)),
        ))
        inner.sim.reset()  # initializes the camera added after env creation
    spec = inner.robot_spec
    device = inner.device
    print(f"[validate] robot {spec.name}: {spec.urdf_path}")

    obs, _ = env.reset()
    lab_names = list(inner.robot.joint_names)
    arm_ids = list(inner._arm_joint_ids)

    def lab_target(arm_pos: tuple[float, ...]) -> torch.Tensor:
        named = spec.arm_joint_dict(arm_pos)
        target = torch.tensor([named.get(n, 0.0) for n in lab_names], device=device)
        return target.repeat(args.num_envs, 1)

    def set_state(q: torch.Tensor) -> None:
        inner.robot.write_joint_position_to_sim_index(position=q)
        inner.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        inner._prev_targets[:] = q
        inner._cur_targets[:] = q

    zero_action = torch.zeros(args.num_envs, inner.cfg.action_space, device=device)

    # --- 1. Palm center vs URDF forward kinematics ---
    urdf = yourdfpy.URDF.load(str(REPO_ROOT / spec.urdf_path), load_meshes=False)
    q_urdf = np.zeros(len(urdf.actuated_joint_names))
    for name, value in spec.arm_joint_dict(spec.arm_default_joint_pos).items():
        q_urdf[urdf.actuated_joint_names.index(name)] = value
    urdf.update_cfg(q_urdf)
    T_base_palm = urdf.get_transform(spec.palm_body_name, urdf.base_link)
    R_base = Rotation.from_quat(spec.base_quat_xyzw).as_matrix()
    expected = np.array(spec.base_pos) + R_base @ (T_base_palm @ np.append(spec.palm_center_offset, 1.0))[:3]
    obs_list = list(cfg.obs.obs_list)
    off = sum(OBS_FIELD_SIZES[f] for f in obs_list[: obs_list.index("palm_pos")])
    observed = obs["policy"][0, off : off + 3].detach().cpu().numpy()
    err_mm = np.linalg.norm(observed - expected) * 1000
    print(f"[validate] 1. palm center obs {observed.round(4)} vs URDF FK {expected.round(4)}: {err_mm:.2f} mm")
    assert err_mm < 2.0, "palm center in sim disagrees with URDF FK"

    # --- 2. Hold default / raised poses ---
    for label, pose in (("default", spec.arm_default_joint_pos),
                        ("start_higher", spec.arm_start_higher_joint_pos)):
        target = lab_target(pose)
        set_state(target)
        inner._replay_target_lab_order = target
        max_dev = torch.zeros(len(lab_names), device=device)
        for _ in range(120):
            env.step(zero_action)
            dev = (inner.robot.data.joint_pos.torch - target).abs().amax(dim=0)
            max_dev = torch.maximum(max_dev, dev)
        worst = int(max_dev.argmax())
        print(f"[validate] 2. hold {label}: max joint deviation {max_dev.max().item():.2e} rad "
              f"({lab_names[worst]})")
        assert max_dev.max().item() < HOLD_TOL_RAD, f"robot drifts from the {label} pose"

    # --- 3. Arm step response ---
    print(f"[validate] 3. arm step response (+{STEP_RAD} rad, 1 s):")
    print("      joint              overshoot  rise90 (s)  final err (rad)")
    default = lab_target(spec.arm_default_joint_pos)
    for lab_j in arm_ids:
        set_state(default)
        target = default.clone()
        target[:, lab_j] += STEP_RAD
        inner._replay_target_lab_order = target
        traj = []
        for _ in range(60):
            env.step(zero_action)
            traj.append(inner.robot.data.joint_pos.torch[0, lab_j].item() - default[0, lab_j].item())
        traj = np.array(traj)
        overshoot = max(0.0, traj.max() - STEP_RAD) / STEP_RAD
        reached = np.nonzero(traj >= 0.9 * STEP_RAD)[0]
        rise = (reached[0] + 1) / 60.0 if len(reached) else float("nan")
        final_err = abs(traj[-10:].mean() - STEP_RAD)
        print(f"      {lab_names[lab_j]:<18} {overshoot:9.1%}  {rise:10.3f}  {final_err:15.2e}")
        assert final_err < 0.01 and overshoot < 0.5, f"{lab_names[lab_j]} does not settle"

    # --- 4. Random actions ---
    inner._replay_target_lab_order = None
    set_state(default)
    max_arm_vel = 0.0
    for _ in range(args.random_steps):
        action = 2.0 * torch.rand_like(zero_action) - 1.0
        obs, *_ = env.step(action)
        assert torch.isfinite(obs["policy"]).all(), "non-finite observation"
        max_arm_vel = max(max_arm_vel, inner.robot.data.joint_vel.torch[:, arm_ids].abs().max().item())
    print(f"[validate] 4. {args.random_steps} random-action steps finite; max |arm qd| {max_arm_vel:.2f} rad/s")
    assert max_arm_vel < 10.0, "arm joint velocity blew up"

    # --- 5. Renders ---
    if args.render:
        import imageio.v2 as imageio

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        set_state(default)
        inner._replay_target_lab_order = default
        origins = inner.scene.env_origins
        views = {"front": ((0.9, -1.0, 1.1), (0.0, 0.4, 0.6)), "side": ((1.6, 0.4, 0.9), (0.0, 0.4, 0.6))}
        for view, (eye, lookat) in views.items():
            camera.set_world_poses_from_view(
                origins + torch.tensor(eye, device=device),
                origins + torch.tensor(lookat, device=device),
            )
            for _ in range(30):
                env.step(zero_action)
            camera.update(inner.step_dt)
            frame = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
            path = out_dir / f"{spec.name}_{view}.png"
            imageio.imwrite(path, frame)
            print(f"[validate] 5. wrote {path}")

    print(f"[validate] robot spec validation OK ({spec.name})")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
