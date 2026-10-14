"""Generate the Flexiv Rizon 4 + left Sharpa HA4 URDF.

The arm comes from Flexiv's official Rizon 4 parameters, vendored in
``assets/urdf/rizon4_sharpa_description/flexiv_rizon4`` (reproducing what
``flexiv_description``'s ``flexiv_arm.xacro`` emits, without needing ROS). The
hand is the ``sharpa_mount`` subtree of the KUKA + Sharpa URDF, copied verbatim
except for the adapter offset, so both robots share the exact same hand model.

    .venv_isaacsim/bin/python scripts/generate_rizon4_sharpa_urdf.py

Re-run after changing FLANGE_TO_SHARPA_MOUNT_* or SHARPA_MOUNT_TO_HAND_XYZ below, then update
``simtoolreal_shared/robots.py`` (palm offset, default poses) to match.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEXIV_DIR = REPO_ROOT / "assets/urdf/rizon4_sharpa_description/flexiv_rizon4"
KUKA_SHARPA_URDF = (
    REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)
OUTPUT_URDF = REPO_ROOT / "assets/urdf/rizon4_sharpa_description/rizon4_left_sharpa.urdf"

PREFIX = "rizon4_"

# Fixed transform from the Rizon 4 flange frame to the Sharpa ``sharpa_mount``
# link. The copied hand subtree adds sharpa_mount -> left_hand_C_MC (yaw -90 deg,
# as on the KUKA), with its offset replaced by SHARPA_MOUNT_TO_HAND_XYZ below.
# PLACEHOLDER: the yaw of the hand about the flange axis is still the KUKA's;
# confirm it on the real robot and regenerate.
FLANGE_TO_SHARPA_MOUNT_XYZ = (0.0, 0.0, 0.0)
FLANGE_TO_SHARPA_MOUNT_RPY = (0.0, 0.0, 0.0)

# Offset of the hand base (left_hand_C_MC) along the flange axis. The KUKA URDF
# uses 0.05 m for its adapter; the real Rizon + Sharpa adapter is 5 mm.
SHARPA_MOUNT_TO_HAND_XYZ = (0.0, 0.0, 0.005)

# Sharpa meshes stay in the KUKA description directory; reference them relatively.
HAND_MESH_DIR_REWRITE = ("left_sharpa_meshes/", "../kuka_sharpa_description/left_sharpa_meshes/")


def _fmt(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def _origin(parent: ET.Element, xyz, rpy) -> None:
    ET.SubElement(parent, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))


def _load_section(name: str, key: str) -> dict:
    with open(FLEXIV_DIR / "config" / name) as f:
        return yaml.safe_load(f)[key]


def _arm_link(name: str, inertia: dict, mesh_stem: str) -> ET.Element:
    link = ET.Element("link", name=PREFIX + name)

    inertial = ET.SubElement(link, "inertial")
    o = inertia["origin"]
    _origin(inertial, (o["x"], o["y"], o["z"]), (o["roll"], o["pitch"], o["yaw"]))
    ET.SubElement(inertial, "mass", value=repr(float(inertia["mass"])))
    i = inertia["inertia"]
    ET.SubElement(
        inertial, "inertia",
        **{k: repr(float(i[k])) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")},
    )

    visual = ET.SubElement(link, "visual")
    _origin(visual, (0, 0, 0), (0, 0, 0))
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh",
                  filename=f"flexiv_rizon4/meshes/visual/{mesh_stem}.obj")

    collision = ET.SubElement(link, "collision")
    _origin(collision, (0, 0, 0), (0, 0, 0))
    ET.SubElement(ET.SubElement(collision, "geometry"), "mesh",
                  filename=f"flexiv_rizon4/meshes/collision/{mesh_stem}.stl")
    return link


def _fixed_joint(name: str, parent: str, child: str, xyz, rpy) -> ET.Element:
    joint = ET.Element("joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    _origin(joint, xyz, rpy)
    return joint


def build_arm() -> list[ET.Element]:
    kinematics = _load_section("default_kinematics.yaml", "kinematics")
    inertias = _load_section("inertia_parameters.yaml", "inertia_parameters")
    limits = _load_section("joint_limits.yaml", "joint_limits")

    elements = [_arm_link("base_link", inertias["base_link"], "link0")]
    parent = PREFIX + "base_link"
    for idx in range(1, 8):
        kin = kinematics[f"joint{idx}"]
        lim = limits[f"joint{idx}"]
        joint = ET.Element("joint", name=f"{PREFIX}joint{idx}", type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=f"{PREFIX}link{idx}")
        _origin(joint, (kin["x"], kin["y"], kin["z"]), (kin["roll"], kin["pitch"], kin["yaw"]))
        ET.SubElement(joint, "axis", xyz=kin.get("axis", "0 0 1"))
        # Soft limits: the controller stops the real arm there, so the policy must
        # never target beyond them. Hard limits are ~5 deg further out.
        ET.SubElement(
            joint, "limit",
            lower=repr(float(lim["soft_lower"])), upper=repr(float(lim["soft_upper"])),
            effort=repr(float(lim["effort"])), velocity=repr(float(lim["velocity"])),
        )
        elements += [joint, _arm_link(f"link{idx}", inertias[f"link{idx}"], f"link{idx}")]
        parent = f"{PREFIX}link{idx}"

    flange = kinematics["link7_to_flange"]
    elements += [
        _fixed_joint(
            f"{PREFIX}link7_to_flange", f"{PREFIX}link7", f"{PREFIX}flange",
            (flange["x"], flange["y"], flange["z"]),
            (flange["roll"], flange["pitch"], flange["yaw"]),
        ),
        ET.Element("link", name=f"{PREFIX}flange"),
        _fixed_joint(
            f"{PREFIX}flange_to_sharpa_mount", f"{PREFIX}flange", "sharpa_mount",
            FLANGE_TO_SHARPA_MOUNT_XYZ, FLANGE_TO_SHARPA_MOUNT_RPY,
        ),
    ]
    return elements


def extract_hand(kuka_root: ET.Element) -> list[ET.Element]:
    """Return the links and joints of the subtree rooted at ``sharpa_mount``."""
    links = {link.get("name"): link for link in kuka_root.findall("link")}
    joints = kuka_root.findall("joint")
    keep = {"sharpa_mount"}
    changed = True
    while changed:
        changed = False
        for joint in joints:
            if joint.find("parent").get("link") in keep and joint.find("child").get("link") not in keep:
                keep.add(joint.find("child").get("link"))
                changed = True

    elements = []
    for element in kuka_root:
        is_hand_link = element.tag == "link" and element.get("name") in keep
        is_hand_joint = element.tag == "joint" and element.find("parent").get("link") in keep
        if is_hand_link or is_hand_joint:
            elements.append(copy.deepcopy(element))
    assert {e.get("name") for e in elements if e.tag == "link"} == keep
    assert len(keep) < len(links)

    mount_joint = next(e for e in elements if e.tag == "joint" and e.get("name") == "sharpa_mount_joint")
    mount_joint.find("origin").set("xyz", _fmt(SHARPA_MOUNT_TO_HAND_XYZ))

    old, new = HAND_MESH_DIR_REWRITE
    for mesh in (m for e in elements for m in e.iter("mesh")):
        filename = mesh.get("filename")
        assert filename.startswith(old), filename
        mesh.set("filename", new + filename[len(old):])
    return elements


def main() -> None:
    kuka_root = ET.parse(KUKA_SHARPA_URDF).getroot()

    robot = ET.Element("robot", name="rizon4_sharpa")
    for material in kuka_root.findall("material"):
        robot.append(copy.deepcopy(material))
    for element in build_arm() + extract_hand(kuka_root):
        robot.append(element)

    ET.indent(robot, space="  ")
    header = (
        '<?xml version="1.0"?>\n'
        "<!-- Generated by scripts/generate_rizon4_sharpa_urdf.py. Do not edit by hand. -->\n"
    )
    OUTPUT_URDF.write_text(header + ET.tostring(robot, encoding="unicode") + "\n")

    n_revolute = sum(1 for j in robot.findall("joint") if j.get("type") == "revolute")
    print(f"wrote {OUTPUT_URDF.relative_to(REPO_ROOT)} "
          f"({len(robot.findall('link'))} links, {n_revolute} revolute joints)")


if __name__ == "__main__":
    main()
