# π0-FAST old-name public-config bridge

This is the post-result V3-A002 recovery gate. It is a separate cohort, not an
exact reconstruction of the missing historical repositories.

The bridge uses public OpenPI commit `235044e`, whose registered config name
and semantics match the committed historical readiness record:
`pi0_fast_droid_jointpos`, `SimpleDataConfig`, 10×8 action chunks, and the
default 250-token FAST context. It uses the already hashed π0-FAST DROID
checkpoint and checkpoint-local DROID normalization statistics.

The public commit postdates the checkpoint and is not the missing historical
runtime. It isolates the old registered config semantics, principally the
250-token FAST context, after the V3-A001 180-token probe failed.

The three-request fixed-observation gate must pass before any simulator cell.
Historical, V3-A001, and V3-A002 evidence are always reported separately.
