"""Derive the Flexiv Rizon 4 values in ``simtoolreal_shared/robots.py`` from the KUKA ones.

- palm_center_offset: the same physical point on the hand as the KUKA's palm
  center, re-expressed in the Rizon palm body (``rizon4_link7``) frame.
- arm_default_joint_pos / arm_start_higher_joint_pos: inverse kinematics that put
  the hand (``left_hand_C_MC``) at the same world pose as the KUKA's default and
  raised poses. Among the default-pose solutions at least LIMIT_MARGIN_DEG
  inside the limits, take the least twisted one (joints 3 and 5 nearest zero,
  like the KUKA default); the raised pose is the solution nearest the default.
- With ``--reachability``: a comparison of both arms over the goal sampling volume.

    .venv_isaacsim/bin/python scripts/compute_rizon4_sharpa_defaults.py [--reachability]

Re-run after regenerating the URDF (e.g. with a measured mount) and paste the
printed values into FLEXIV_RIZON4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yourdfpy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_shared.robots import FLEXIV_RIZON4, KUKA_IIWA14, RobotSpec  # noqa: E402

HAND_FRAME = "left_hand_C_MC"
LIMIT_MARGIN_DEG = 10.0
# Goal sampling volume in the env frame (ResetCfg.target_volume_mins/maxs).
TARGET_VOLUME_MINS = np.array([-0.35, -0.2, 0.6])
TARGET_VOLUME_MAXS = np.array([0.35, 0.2, 0.95])
RNG = np.random.default_rng(0)


class ArmChain:
    """Numpy FK of a serial arm from its base link to one frame of the URDF."""

    def __init__(self, spec: RobotSpec, frame: str):
        self.spec = spec
        urdf = yourdfpy.URDF.load(str(REPO_ROOT / spec.urdf_path), load_meshes=False)
        by_child = {joint.child: joint for joint in urdf.robot.joints}
        chain, link = [], frame
        while link != urdf.base_link:
            chain.append(by_child[link])
            link = chain[-1].parent
        self.chain = chain[::-1]
        # (origin, unit axis or None) per joint, precomputed for fast FK.
        self._steps = [
            (np.asarray(j.origin, dtype=float),
             np.asarray(j.axis, dtype=float) / np.linalg.norm(j.axis) if j.type == "revolute" else None)
            for j in self.chain
        ]
        revolute = [j for j in self.chain if j.type == "revolute"]
        assert tuple(j.name for j in revolute) == spec.arm_joint_names
        self.lower = np.array([j.limit.lower for j in revolute])
        self.upper = np.array([j.limit.upper for j in revolute])
        self.T_world_base = np.eye(4)
        self.T_world_base[:3, 3] = spec.base_pos
        self.T_world_base[:3, :3] = Rotation.from_quat(spec.base_quat_xyzw).as_matrix()

    def fk(self, q: np.ndarray) -> np.ndarray:
        """World pose of the chain's frame."""
        T = self.T_world_base.copy()
        qi = iter(q)
        for origin, axis in self._steps:
            T = T @ origin
            if axis is not None:
                # Rodrigues' formula.
                angle = next(qi)
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R = np.eye(4)
                R[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                T = T @ R
        return T

    def ik(self, T_target: np.ndarray, seeds, *, position_only=False, margin_deg=LIMIT_MARGIN_DEG,
           max_solutions: int | None = None):
        """Return converged solutions within the (shrunk) limits, best margin first."""
        margin = np.radians(margin_deg)
        lo, hi = self.lower + margin, self.upper - margin

        def residual(q):
            T = self.fk(q)
            pos_err = T[:3, 3] - T_target[:3, 3]
            if position_only:
                return pos_err
            rot_err = Rotation.from_matrix(T_target[:3, :3].T @ T[:3, :3]).as_rotvec()
            return np.concatenate([pos_err, 0.2 * rot_err])

        solutions = []
        for seed in seeds:
            result = least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
            if np.linalg.norm(result.fun) < 1e-5:
                solutions.append(result.x)
                if max_solutions is not None and len(solutions) >= max_solutions:
                    break
        return sorted(solutions, key=lambda q: -self.limit_margin_deg(q))

    def ik_nearest(self, T_target: np.ndarray, q_ref: np.ndarray, weight: float = 1e-3) -> np.ndarray:
        """Hand-pose solution closest to q_ref in joint space (uses the redundancy)."""
        def residual(q):
            T = self.fk(q)
            rot_err = Rotation.from_matrix(T_target[:3, :3].T @ T[:3, :3]).as_rotvec()
            return np.concatenate([T[:3, 3] - T_target[:3, 3], 0.2 * rot_err, weight * (q - q_ref)])

        q = least_squares(residual, q_ref, bounds=(self.lower, self.upper), xtol=1e-14, ftol=1e-14).x
        pose_err = np.linalg.norm(residual(q)[:6])
        if pose_err > 1e-4:
            raise SystemExit(f"IK near {q_ref} did not converge (pose error {pose_err:.2e})")
        return q

    def limit_margin_deg(self, q: np.ndarray) -> float:
        return float(np.degrees(np.min(np.minimum(q - self.lower, self.upper - q))))

    def random_seeds(self, n: int) -> list[np.ndarray]:
        return [self.lower + (self.upper - self.lower) * RNG.random(7) for _ in range(n)]


def palm_center_offset(kuka_palm: ArmChain, rizon_palm: ArmChain, kuka_hand: ArmChain,
                       rizon_hand: ArmChain) -> np.ndarray:
    """Re-express the KUKA palm-center offset in the Rizon palm body frame."""
    q0 = np.zeros(7)
    T_palm_hand_k = np.linalg.inv(kuka_palm.fk(q0)) @ kuka_hand.fk(q0)
    T_palm_hand_r = np.linalg.inv(rizon_palm.fk(q0)) @ rizon_hand.fk(q0)
    p_hand = np.linalg.inv(T_palm_hand_k) @ np.append(KUKA_IIWA14.palm_center_offset, 1.0)
    return (T_palm_hand_r @ p_hand)[:3]


def reachability(chain: ArmChain, offset, T_reference: np.ndarray, *, fixed_orientation: bool) -> float:
    """Fraction of goal-volume grid points the palm center can reach."""
    grid = np.stack(np.meshgrid(*[np.linspace(a, b, n) for a, b, n in
                                  zip(TARGET_VOLUME_MINS, TARGET_VOLUME_MAXS, (8, 5, 5))],
                                indexing="ij"), -1).reshape(-1, 3)
    T_offset = np.eye(4)
    T_offset[:3, 3] = offset
    reached = 0
    for point in grid:
        # Target the palm body pose that puts the palm center on the grid point.
        T_palm_center = T_reference @ T_offset
        T_palm_center[:3, 3] = point
        T_target = T_palm_center @ np.linalg.inv(T_offset)
        seeds = [chain.spec.arm_default_joint_pos, *chain.random_seeds(8)]
        if fixed_orientation:
            solutions = chain.ik(T_target, seeds, margin_deg=0.0, max_solutions=1)
        else:
            # Position-only IK of the palm center itself.
            solutions = []
            for seed in seeds:
                result = least_squares(
                    lambda q: (chain.fk(q) @ np.append(offset, 1.0))[:3] - point,
                    seed if np.all((seed >= chain.lower) & (seed <= chain.upper)) else
                    np.clip(seed, chain.lower, chain.upper),
                    bounds=(chain.lower, chain.upper),
                )
                if np.linalg.norm(result.fun) < 5e-3:
                    solutions.append(result.x)
                    break
        reached += bool(solutions)
    return reached / len(grid)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    kuka_hand, rizon_hand = ArmChain(KUKA_IIWA14, HAND_FRAME), ArmChain(FLEXIV_RIZON4, HAND_FRAME)
    kuka_palm = ArmChain(KUKA_IIWA14, KUKA_IIWA14.palm_body_name)
    rizon_palm = ArmChain(FLEXIV_RIZON4, FLEXIV_RIZON4.palm_body_name)

    offset = palm_center_offset(kuka_palm, rizon_palm, kuka_hand, rizon_hand)

    flexiv_home = np.radians([0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0])
    T_default = kuka_hand.fk(np.array(KUKA_IIWA14.arm_default_joint_pos))
    default_solutions = rizon_hand.ik(
        T_default, [flexiv_home, *rizon_hand.random_seeds(300)], max_solutions=80
    )
    if not default_solutions:
        raise SystemExit("No Rizon IK solution for the KUKA default hand pose")
    q_default = min(default_solutions, key=lambda q: q[2] ** 2 + q[4] ** 2)

    # Like the KUKA's (two joints moved by 10 deg), the raised pose should stay
    # close to the default: take the raised hand-pose solution nearest to it.
    T_higher = kuka_hand.fk(np.array(KUKA_IIWA14.arm_start_higher_joint_pos))
    q_higher = rizon_hand.ik_nearest(T_higher, q_default)

    # Sanity: palm centers coincide in the world for both poses.
    for name, q_k, q_r in (("default", KUKA_IIWA14.arm_default_joint_pos, q_default),
                           ("start_higher", KUKA_IIWA14.arm_start_higher_joint_pos, q_higher)):
        p_k = kuka_palm.fk(np.array(q_k)) @ np.append(KUKA_IIWA14.palm_center_offset, 1.0)
        p_r = rizon_palm.fk(q_r) @ np.append(offset, 1.0)
        print(f"{name}: palm center KUKA {p_k[:3]}  Rizon {p_r[:3]}  "
              f"|diff| {np.linalg.norm(p_k[:3] - p_r[:3]) * 1000:.2f} mm  "
              f"Rizon limit margin {rizon_hand.limit_margin_deg(q_r):.1f} deg")
    print(f"raised - default (deg): {np.degrees(q_higher - q_default)}")

    print(f"\nIK: {len(default_solutions)} default-pose solutions with >= {LIMIT_MARGIN_DEG:.0f} deg "
          f"limit margin; picked the least twisted one.")
    if "--reachability" in sys.argv:  # slow: ~10 min
        print("\nReachability of the goal volume (palm center, 200 grid points):")
        for label, fixed in (("any orientation", False), ("default hand orientation", True)):
            r_k = reachability(kuka_palm, np.array(KUKA_IIWA14.palm_center_offset),
                               kuka_palm.fk(np.array(KUKA_IIWA14.arm_default_joint_pos)),
                               fixed_orientation=fixed)
            r_r = reachability(rizon_palm, offset, rizon_palm.fk(q_default), fixed_orientation=fixed)
            print(f"  {label:>24}: KUKA {r_k:6.1%}   Rizon 4 {r_r:6.1%}")

    fmt = lambda v: "(" + ", ".join(f"{x:.4f}" for x in v) + ")"  # noqa: E731
    print("\nFLEXIV_RIZON4 values:")
    print(f"    arm_default_joint_pos={fmt(q_default)},")
    print(f"    arm_start_higher_joint_pos={fmt(q_higher)},")
    print(f"    palm_center_offset={fmt(offset)},")


if __name__ == "__main__":
    main()
