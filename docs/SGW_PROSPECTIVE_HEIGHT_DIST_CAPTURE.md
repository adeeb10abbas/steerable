# Prospective HEIGHT/DIST native capture

`prospective_family_scene.py` creates a new **prospective** USD overlay that
sublayers the supplied base scene without modifying it. Its supports and the
DIST plate are source-controlled design inputs, not measured geometry,
qualified fixtures, or frozen candidates. No command below creates candidates
or consumes the family’s 100-candidate cap.

The overlay builder binds the exact base scene bytes and the existing measured
LAT workspace receipt. For the current known baseline the receipt is
`artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json`,
whose receipt SHA-256 is
`3766dd5b9ed08f7fc60525077ba96df0d98b67e980ff453ee7d19559932b8239`;
the base asset manifest hash is
`3a9b8ceeff333aa3a060dad97707b5f2a50e7c4db1423c84350bf4d8a19b0806`.

The builder requires that receipt's **v2 measurement schema** and authors the
measured cube/bowl orientations as well as their translated actor roots.
Both families have a separate neutral cube support. DIST uses a circular plate,
with its support top at the disc's bottom, and separate landing supports clear
of the anchors. Both counterbalances are checked against the actual committed
workspace geometry; these are authoring checks, not physical qualification.

Parent must use the pinned RoboLab `0aef241` base USD path below on the assigned
lane only after reviewing the generated manifest:

```bash
.venv/bin/python -m experiments.workshops.spatial_grounding_v1.prospective_family_scene \
  --family HEIGHT --upper-side left \
  --base-scene /data/users/ali/vla_wam/external/RoboLab-pi05-v3-0aef241/assets/scenes/rubiks_cube_banana_bowl.usda \
  --base-workspace-receipt artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json \
  --output-usda /data/users/ali/sgw-01/prospective/height-left.usda \
  --manifest-output /data/users/ali/sgw-01/prospective/height-left.manifest.json

.venv/bin/python -m experiments.workshops.spatial_grounding_v1.prospective_family_capture \
  --study-root "$PWD" --robolab-root /ABSOLUTE/PINNED/RoboLab \
  --assets-manifest /ABSOLUTE/assets-manifest.json \
  --renderer-receipt /ABSOLUTE/passed-renderer.json \
  --overlay-manifest /data/users/ali/sgw-01/prospective/height-left.manifest.json \
  --output /data/users/ali/sgw-01/prospective/height-left/capture.json \
  --headless --renderer realtime --rendering-type balanced --rendering_mode balanced \
  --device cuda:0 --num-envs 1
```

Run DIST analogously with `--family DIST --bowl-side left|right`. The capture
requires a 120-frame render-only warmup/video, zero controller actions, actual
root/center/offset/AABB records, lossless camera arrays, and contact sensor
inventory. Before candidate proposals, a parent-reviewed capture must prove
the named supports have usable cube-contact sensors, each final support is
reachable and released, the plate is visually distinct in actual views, and
the counterbalanced neutral starts are physically valid.
