# VLA/WAM video map

Start with the [interactive gallery](../../../docs/VLA_WAM_STEERABILITY_VIDEO_GALLERY.html). This directory is the complete inventory of committed MP4s; the gallery is the curated reading order.

## How to read the media

- **Actual rollout** means simulator execution and may support behavioral outcomes.
- **Model prediction / imagination** is generated future evidence, never an executed episode.
- **Prediction-only interface probe** has no controller rollout and no success score.
- **Alternate square encodes and reconstruction components** are presentation/support files, not extra evidence.

DROID/RoboLab and RoboTwin use different tasks and denominators. Never pool their success rates.

## Model-level inventory

| Class | Model | Arena | Canonical execution | Canonical prediction | Archive / support |
| --- | --- | --- | ---: | ---: | ---: |
| VLA | GR00T N1.7 DROID | DROID / RoboLab | 1 | 0 | 0 |
| VLA | LingBot-VLA 4B | RoboTwin place-A-relative-to-B | 1 | 0 | 0 |
| VLA | π0-FAST DROID — historical frozen-stack reference | DROID / RoboLab | 1 | 0 | 1 |
| VLA | π0.5 DROID — current-stack V2-A010 | DROID / RoboLab | 1 | 0 | 0 |
| WAM | Cosmos3 Edge DROID | DROID / RoboLab | 1 | 0 | 0 |
| WAM | Cosmos3 Edge base — DROID | DROID / RoboLab | 0 | 1 | 0 |
| WAM | Cosmos3 Nano Policy DROID — V2-A011 | DROID / RoboLab | 1 | 1 | 0 |
| WAM | Cosmos3-Super base | DROID / RoboLab conditioning image only | 0 | 1 | 0 |
| WAM | DreamZero DROID | DROID / RoboLab | 3 | 3 | 9 |
| WAM | Efficient-WAM-RT | RoboTwin place-A-relative-to-B | 2 | 0 | 1 |
| WAM | FastWAM | RoboTwin place-A-relative-to-B | 1 | 0 | 3 |
| WAM | Light-WAM | RoboTwin place-A-relative-to-B | 1 | 0 | 0 |
| WAM | LingBot-VA | RoboTwin place-A-relative-to-B | 2 | 0 | 1 |

## Machine-readable inventory

- [Complete CSV](media_catalog.csv): one row per committed MP4.
- [Hash-bearing JSON](media_catalog.json): roles, sizes, SHA-256 digests, and source manifests.
- [Gallery manifest](video_first_gallery_manifest.json): canonical publication selections.

Catalog total: **36 committed MP4s**. A file count is not an episode count.
