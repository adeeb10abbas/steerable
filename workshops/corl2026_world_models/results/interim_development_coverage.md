# Development recording and timing coverage

This receipt-level audit rechecks 11 compact published files at results commit
`12839900e5c46299e11fff4813e8602949055b86`: eight valid block aggregates, both
timing sidecars, and the formal development compiler receipt. It does not
revalidate the raw PVC corpus or establish current cluster process state.

| Model | Development base-layout pairs | Valid cells | Executed actions | Model requests |
| --- | ---: | ---: | ---: | ---: |
| N3 | 4 | 16 | 7,200 | 240 |
| D1 | 4 | 16 | 7,200 | 912 |

These are completed development episodes, not software tests. They exclude the
eight valid pilot cells and all technical-invalid attempts preserved in the
execution ledger. They are not a task-success or forecast-accuracy result.

D1's timing evidence partitions its 912 requests into 240 source-proven full
decodes and 672 incremental decodes without a supported physical-time mapping.
Of the 240 full decodes, 224 have native-clock matches and 16 are truncated by
the available executed-action prefix. Thus 224/912 (24.56%) of all development
requests have clock matches, contributing 448 target bindings. The 672 unmapped
requests and 32 prefix-truncated target bindings remain missing; neither is
assigned zero prediction error. A complete-case score would describe only the
eligible subset, not the full D1 request population.

N3's sidecar references all 240 development requests. This compact audit does
not derive target-level coverage from that request count; full target records
remain on the PVC. Camera-crop/alignment qualification is a separate gate for
both models.

Forecast error, baseline skill and their missingness bounds remain unavailable
without signed alignment and legitimate independent human labels/consensus.
No movement threshold, label, score or confirmation release is inferred here.
All 24 planned confirmation layout pairs remain retained, with 0/192
confirmation episodes released at the last reconciled scientific snapshot.

The machine-readable companion records exact Git paths, byte counts and hashes.
Reproduce it from this checkout without network, raw-data download or writes:

```bash
python3 workshops/corl2026_world_models/analysis/audit_published_development_coverage.py \
  --check workshops/corl2026_world_models/results/interim_development_coverage.json
```

The pinned results commit must already be present locally. This audit creates
zero new model requests, actions or behavioral episodes.
