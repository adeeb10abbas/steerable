# SGW HEIGHT/DIST capture requirements

HEIGHT and DIST have no approved fixture slots or candidate files. The LAT
workspace receipt must not be repurposed: it contains neither elevated support
geometry nor a plate.

Before `height_dist_proposals.py` can materialize an explicitly unqualified
candidate, a zero-model native capture must provide a hash-bound workspace
receipt containing:

- immutable workspace receipt, asset-manifest hash, task asset, and a native
  scene asset/object inventory;
- cube/bowl roots, WXYZ orientations, scoring centers, and root-local center
  offsets for every layout; DIST additionally requires the plate;
- HEIGHT: named, immutable higher and lower released support surfaces,
  their actual cube-contact sensor IDs, measured cube-center landing coordinates, an intermediate-height bowl, and
  the `upper_support_side` label;
- DIST: named, immutable near-bowl and near-plate released support surfaces,
  their actual cube-contact sensor IDs, measured cube-center landing coordinates, bowl/plate scoring centers, and
  the `bowl_side` label;
- for both families: contact instrumentation proving each proposed final
  support, collision geometry/asset provenance, camera/reset identity, and a
  measured Abs-IK calibration before the six scripted physical checks.

The proposal code validates neutrality, both requested 30 mm margins, and the
counterbalance labels from these measurements. It does **not** create supports,
plate geometry, collision bounds, root poses, or controller actions. A missing
field is a qualification blocker, not a default.
