# Flexiv Rizon 4 description (vendored)

Kinematics, inertia and joint-limit parameters (`config/`) and meshes (`meshes/`) for the
Flexiv Rizon 4, copied from
[flexivrobotics/flexiv_description](https://github.com/flexivrobotics/flexiv_description)
at commit `77dd8d8ec4c48a06778ac3886569c9d24e7a977a`:

- `config/Rizon4/{default_kinematics,inertia_parameters,joint_limits,initial_positions}.yaml` → `config/`
- `meshes/Rizon4/visual/link*.{obj,mtl}` → `meshes/visual/`
- `meshes/Rizon4/collision/link*.stl` → `meshes/collision/`

One change from upstream: `meshes/visual/link4.obj` had a corrupt face line
(`f 2756//2756 2755//2755 2`, a 10 cm sliver whose third vertex lacks a normal index),
which Isaac Sim 6's URDF importer rejects, dropping the whole mesh. That face is removed.

Licensed under Apache-2.0; see `assets/licenses/flexiv_description-LICENSE.txt`.

`scripts/generate_rizon4_sharpa_urdf.py` builds `../rizon4_left_sharpa.urdf` from these files.
