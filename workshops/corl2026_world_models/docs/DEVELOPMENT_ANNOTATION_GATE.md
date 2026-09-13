# Development annotation gate operations

`development_annotation_gate_jobs.py` is the bounded operational bridge between
the prepared development media and the existing human annotation workflow. It
does not perform annotation, infer a movement threshold, release confirmation,
or contact a rater. Its exact machine-readable policy is
`development_annotation_gate_contract.json`.

## Inputs that must come from outside the job

Private packet packaging remains ineligible until all of these exact artifacts
exist:

1. the terminal successful `development_annotation_media_jobs.py` receipt and
   its recoverable preparation tree on the GM PVC;
2. a real human-authored, hash-bound
   `wmf-forecast-pixel-blindness-review-v1` receipt for the exact rendered
   hashes;
3. the development annotation freeze, whose rubric, illustrated examples, and
   separate example pixel-review receipt pass `validate_freeze`; and
4. later, one externally authored, locked response per delivered batch.

The job never fills a response template, chooses a reviewer/rater/adjudicator,
or converts a machine-generated label into human evidence. Operator and
collector identifiers passed to the commands are audit identities, stored only
as SHA-256 digests; they are not substitutes for the human attestations inside
the supplied artifacts.

As elsewhere in this workshop, `payload_sha256` is an integrity checksum, not a
cryptographic identity signature or authorization credential. Authenticating
the people and the external delivery route remains an operator responsibility.

## Private packaging

`package-private` authenticates the exact terminal media-job receipt through
the media queue's existing terminal validator. It then calls, rather than
reimplements:

- `prepare_development_annotation_media.finalize_reviewed_inventory`;
- `forecast_annotation_workflow.validate_freeze`; and
- `forecast_annotation_workflow.package_packets`.

The resulting packet tree and restricted identity map remain below the
caller-supplied private root. The root is mode `0700`, the restricted map is
mode `0600`, and `package_receipt.json` is written last. A failed attempt has no
terminal package receipt and is ineligible for release; its partial bytes are
preserved for diagnosis.

Example, only after the human artifacts exist:

```bash
python workshops/corl2026_world_models/experiments/forecast_layout/development_annotation_gate_jobs.py \
  package-private \
  --study-commit COMMIT40 \
  --media-job-receipt /absolute/media_job_receipt.json \
  --media-job-receipt-sha256 SHA256 --media-job-receipt-bytes BYTES \
  --pixel-blindness-receipt /absolute/human_review.json \
  --pixel-blindness-receipt-sha256 SHA256 --pixel-blindness-receipt-bytes BYTES \
  --development-freeze /absolute/development_freeze.json \
  --development-freeze-sha256 SHA256 --development-freeze-bytes BYTES \
  --private-root /absolute/private/development_annotation_gate
```

## One-batch lifecycle

The handoff root is a dedicated controlled filesystem location, not the Git
results branch. Its only permitted states are empty or one directory named
`active`. A release copies one existing blinded batch into a temporary sibling,
adds its signed delivery receipt, revalidates the copied bytes, and exposes the
whole tree with one atomic directory rename. Any extra file, another active
batch, a skipped per-rater ordinal, missing prior collection/revocation, changed
packet byte, symlink, or broken receipt chain fails closed.

Before publication, an immutable private delivery witness is written. It stays
present throughout the active/collected states and is moved into the private
archive at revocation. If the public tree disappears out of band, that witness
blocks re-release instead of allowing the missing tree to masquerade as a
completed revocation.

```bash
python .../development_annotation_gate_jobs.py release-batch \
  --private-root PRIVATE --public-root HANDOFF \
  --slot rater_a --ordinal 1 --released-by authorized-operator
```

The external authorized rater-delivery route may read only `HANDOFF/active`.
It must not receive the private root. The delivery receipt deliberately claims
only publication into this controlled filesystem. It does not claim that an
email, survey, shared drive, or other external route delivered or revoked
access, and it cannot prove that a human did not retain a copy.

After an actual human returns the response, collection requires the caller to
supply its exact path, SHA-256, byte count, and a collector audit code:

```bash
python .../development_annotation_gate_jobs.py collect-response \
  --private-root PRIVATE --public-root HANDOFF \
  --response /absolute/human_response.json \
  --response-sha256 SHA256 --response-bytes BYTES \
  --collector-code authorized-collector
```

Collection invokes the existing exact response and packet validators. It also
requires response start at or after delivery, lock at or before collection,
the sum of per-image annotation seconds to fit inside the session, frozen
per-rater batch order, nonoverlapping sessions, one stable rater identity per
slot, two distinct first-pass raters, and an adjudicator distinct from both.
Operational timestamps more than 300 seconds ahead of the local UTC clock are
also rejected.
The response is copied byte-for-byte into private storage, and its hash and the
collector timestamp are bound in an immutable collection receipt. Collection
will not start if `pending_collections` contains any entry: a partial or complete
temporary directory from an interrupted attempt is evidence requiring explicit
reconciliation, not permission to accept a second response.

Only a collected batch can be revoked:

```bash
python .../development_annotation_gate_jobs.py revoke-delivery \
  --private-root PRIVATE --public-root HANDOFF \
  --revoked-by authorized-operator
```

Revocation atomically removes `HANDOFF/active` and archives its unchanged
packet, human response, and delivery/collection/revocation receipts privately.
Before that removal, the pending root must contain exactly the active delivery
directory and that directory must contain exactly its three expected regular
files. Active and archived inventories likewise enforce their declared regular
file/directory types; FIFOs, sockets, symlinks, and type substitutions fail
before any file is opened or moved. The archive root itself may not be a
symlink, including a dangling one, and the complete prior archive lifecycle is
replayed and bound to the active delivery before revocation moves any live
transaction byte. Foreign files, directories, and crash residue therefore
block revocation without being silently carried past a success receipt. The
same per-record validator used after archival first replays the prospective
delivery, collection, response, frozen packet stream, package and source-free
batch bytes in place, then the same aggregate validator checks the prospective
sorted history for sequence, prior-receipt, time/order, stable-rater and
cross-rater constraints. Nested batch entries are type-checked before any JSON
is opened, so a FIFO or socket cannot stall this preflight.
An interrupted `.revoking` directory blocks later release and must be
reconciled; it is never silently treated as success. The revocation scope is
only the controlled filesystem handoff.

## Adjudication and time accounting

After every first-pass batch for both raters is collected and revoked,
`prepare-adjudication` materializes immutable response sets and calls the
existing `package_adjudication` implementation. Only exact first-pass
disagreements enter the blind adjudicator stream. The same one-active-batch
lifecycle applies to it.

`write-time-accounting` is eligible only after the adjudication packaging
decision and all required adjudicator batches are collected and revoked. It
reports separately:

- `rater_a_total`;
- `rater_b_total`; and
- `adjudicator_total` (exactly zero only when no adjudication was required).

These are sums of human-entered `annotation_seconds`; they are not inferred
from media frames or compute runtime. Every receipt keeps the complete
zero-science vector, `labels_created_by_job: 0`, and
`safe_to_release_confirmation: false`.

## Remaining irreducible human boundary

This code makes the handoff auditable but does not provide a legitimate human
rater channel. A real authorized route must distribute the sole active batch,
collect an independently authored response, and separately account for any
external access/revocation semantics. After both first passes and adjudication,
the existing workflow must still produce final consensus, a development noise
summary, and an explicit authorized measurement-usability decision. Only those
validated artifacts may be used to build a confirmation freeze; this operations
slice never releases confirmation itself.
