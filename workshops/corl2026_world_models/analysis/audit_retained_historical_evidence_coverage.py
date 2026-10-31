#!/usr/bin/env python3
"""Audit complete retained H01--H13 packaging without pooling results.

The registered historical table is distributed across nine grouped analysis
packages.  This audit authenticates the exact retained Git objects, maps every
registered ID exactly once in the grouped-analysis crosswalk, records additional
primary-paper, supplemental, and presentation roles, and checks each package's
activity, gate, and scientific-claim boundaries.  It does not recompute or
combine an outcome, infer a scientific result, recover raw bytes, run a model,
create a label, or change the 232-cell ceiling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


BASE_COMMIT = "432c4bdcbb3b2ebb3ec94aab5ce5455b56f531c6"
BASE_TREE = "68c039ddfe487c2e3e5c978f8b0eb4236ed0dbd0"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
DEFAULT_JSON = PACKAGE / "paper" / "data" / "retained_historical_evidence_coverage.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "retained_historical_evidence_coverage.csv"
PAPER_PATH = "workshops/corl2026_world_models/paper/main.tex"
BUILD_PATH = "workshops/corl2026_world_models/analysis/build_package.py"
PAPER_RESULTS_PATH = "workshops/corl2026_world_models/results/paper_results.json"
PRESENTATION_MEDIA_PATH = "workshops/corl2026_world_models/paper/media/representative_h01_videos.json"

EXPECTED_SOURCE_AGGREGATE = {
    "files": 50,
    "bytes": 2_054_896,
    "lines": 45_726,
    "manifest_sha256": "26dac013bb654e28145e0f2bdfbfe2d8622080c9af84b3773efe5c217a337b38",
}

EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    "workshops/corl2026_world_models/docs/ABLATION_SPEC.md": {"git_blob": "aff0e42e16b4cc542b0f7ff51e2de1dce8b6dadd", "sha256": "fa8cbbac7bf2eff2e01f73150e5cf073f7e4f177d3c0bfd4fc1f41fb3ea9309f", "bytes": 27_094, "lines": 256},
    "workshops/corl2026_world_models/results/existing_ablation_inventory.json": {"git_blob": "31103fe505c836429bd25b5145040e6ef516e708", "sha256": "f8d53f15385220523dac4268c6f0108b98dc7585978f1baafb2e053136507e1b", "bytes": 23_271, "lines": 704},
    "workshops/corl2026_world_models/paper/main.tex": {"git_blob": "60b667dd5cb0919197b26d09b4a9e439d677fd5c", "sha256": "c7fa1c90f09aa73a574799ac11a473c96024b524764779cc423b64853e94155b", "bytes": 12_648, "lines": 108},
    "workshops/corl2026_world_models/analysis/build_package.py": {"git_blob": "e894556081761d5dcc2e4c5978f4dba0cd44a891", "sha256": "06d435de5fbcb7cd00ff00c6ff5272e3b64ae534f0d2524573e497a5b4c15e7f", "bytes": 42_150, "lines": 762},
    "workshops/corl2026_world_models/analysis/extract_paper_results.py": {"git_blob": "73584ac80ad15d37230b8eb682b92a5825c21e66", "sha256": "00e1e6899be38dc4636464206f6de981efe068f95e5d7751dc26fae1c7a53522", "bytes": 19_910, "lines": 338},
    "workshops/corl2026_world_models/results/paper_results.json": {"git_blob": "e780be08d24b22f335625b5266b09e5bd320c714", "sha256": "f4695ae37dfd6b981a133ee10ccd69a2b86988e71d6ff99462bd449189dd94d7", "bytes": 201_220, "lines": 4_869},
    "workshops/corl2026_world_models/tests/test_paper_results.py": {"git_blob": "8df007b812b9ea8f838f5fb9f0d74ab5e02686d3", "sha256": "81ea5ef4a39c57e6be7a7a398d0e2447e26d218c0d1bb835f6c00e46e9c218df", "bytes": 6_118, "lines": 134},
    "workshops/corl2026_world_models/analysis/package_representative_media.py": {"git_blob": "d4e0224e7bd36e639c13a976f79145eb0f75da02", "sha256": "6d370ca17d54c44b2ca164524e4c89ed05355912eca6ed2aa13f5e386b1dddf1", "bytes": 10_707, "lines": 259},
    "workshops/corl2026_world_models/paper/media/representative_h01_videos.json": {"git_blob": "cf16c8176ff66a13fc3dd21df8c7e9380d7eddcf", "sha256": "34f626e91c44f51bdbfd73cfa4ffc48d68fe645d1e85f94e863c56b3451b12ba", "bytes": 8_060, "lines": 201},
    "workshops/corl2026_world_models/tests/test_representative_media_package.py": {"git_blob": "1ae32bbe14ae538961321fb53cb8b1687831ffb5", "sha256": "1b93c6155dc452855b403fa4f669af3bf899a0a37b36cee896cf16d64639d78b", "bytes": 5_308, "lines": 108},
    "workshops/corl2026_world_models/analysis/analyze_h01_requested_region_entry.py": {"git_blob": "a300891a1d2369d9f91e8c58f92abddeca90f640", "sha256": "0230fc77bf6e80fda59b904a799bca9bef77488ae624eaca45611fce34220525", "bytes": 34_585, "lines": 760},
    "workshops/corl2026_world_models/paper/data/h01_requested_region_entry.json": {"git_blob": "d85212111cb07bb92b4f77f83544dc3d12fa3cb4", "sha256": "ff35ff9888ccd86e7bc70970b2f56279fbe90d621f56ea91cfe6069892466152", "bytes": 98_596, "lines": 2_765},
    "workshops/corl2026_world_models/paper/data/h01_requested_region_entry.csv": {"git_blob": "1bfc47d3db02464d826f2ee895010432f748fb54", "sha256": "df715dd4b696a70d93ff0bb7d4520da1d146e8f74c868dcb7fa5dba1759c430f", "bytes": 10_384, "lines": 28},
    "workshops/corl2026_world_models/analysis/analyze_h02_position_reflection.py": {"git_blob": "61000008a18950423080b8045636c7af5170ec82", "sha256": "44b941be8d362a9498798051301a52da99b2679d14f1571f049122fded59f43e", "bytes": 42_021, "lines": 877},
    "workshops/corl2026_world_models/paper/data/h02_position_reflection.json": {"git_blob": "75594e41d3bdb221f0c5cc5080b55fc0a4b00471", "sha256": "9e39905c58cbd674bf41e342cadfd3e97f0c9e618bde8870e1785ed16af6adf1", "bytes": 103_418, "lines": 1_986},
    "workshops/corl2026_world_models/paper/data/h02_position_reflection.csv": {"git_blob": "a3121f0c239a6d15e17eba8d9f9a12d368636c45", "sha256": "8f1a9dd252c80d57fe543707820d3eec0bbd1580bacf61a597d33adc85b9579b", "bytes": 13_277, "lines": 28},
    "workshops/corl2026_world_models/analysis/analyze_h03_h04_guidance_pilots.py": {"git_blob": "a74c0e255bbbe623059440662abb7a6cd141b5f0", "sha256": "08cba5437e3893b3c3e2feff3ba4fc446c301b64d5783126aaf4e42fd92a025e", "bytes": 74_043, "lines": 1_370},
    "workshops/corl2026_world_models/paper/data/h03_h04_guidance_pilots.json": {"git_blob": "5509af6d788585fe5181cea8a2e3d0fb05eed610", "sha256": "ee2b68b5fa931117a9cb7e30c6cef3a9f7d8d60f55f266b411544de68e65e23f", "bytes": 47_579, "lines": 963},
    "workshops/corl2026_world_models/paper/data/h03_h04_guidance_pilots.csv": {"git_blob": "02760fa111b93165821eccda787ff5d8981eb831", "sha256": "6e9b90404ffd0b629bd82e28fabec592485f28ea019d5f42621791e7b430c51f", "bytes": 10_434, "lines": 13},
    "workshops/corl2026_world_models/analysis/analyze_h05_lateral_dose_response.py": {"git_blob": "eb08bb05e1956edd94a30f92725682bf22ce668c", "sha256": "27855ab32cef4a429000dabccbd9de7132f9c979eb7e028b4f83afc842b80ca3", "bytes": 31_211, "lines": 638},
    "workshops/corl2026_world_models/paper/data/h05_lateral_dose_response.json": {"git_blob": "838631f22d1e5529c28ea22f4fcdf9e4990f28dc", "sha256": "754391cf49021912171d6341237d20aa4523a94db22927df6d26d0b0493a2b6c", "bytes": 44_136, "lines": 1_393},
    "workshops/corl2026_world_models/paper/data/h05_lateral_dose_response.csv": {"git_blob": "3901fdaaf92b124f9748fc64c57cfd1222781a15", "sha256": "767b8cc102496407191584894a516f59e6b2baed9ea0ca01aef41dc9890364fd", "bytes": 8_513, "lines": 106},
    "workshops/corl2026_world_models/analysis/analyze_h06_graded_symmetry.py": {"git_blob": "481df82c2683d10d72c37d5b8a4bcabbcc646d27", "sha256": "c9496ec8928953b1ee71dc80202dbf4da80ae3a05503ff44acebbad26c6a3b05", "bytes": 30_670, "lines": 663},
    "workshops/corl2026_world_models/paper/data/h06_graded_symmetry.json": {"git_blob": "bfae8630b88c0528d4d09952d3a490b62d700cae", "sha256": "019bfc090ac928a4d33d3c85ce9050c98e620f4291df10ada27f54ce70f3b89f", "bytes": 51_450, "lines": 1_568},
    "workshops/corl2026_world_models/paper/data/h06_graded_symmetry.csv": {"git_blob": "90bb5a7b59f3d689287d527b87b6769c2a49882f", "sha256": "ad4acd56065d00aa9c6352d651c377e09b805fb8ad23354dd05d4e6fa651f099", "bytes": 9_932, "lines": 109},
    "workshops/corl2026_world_models/analysis/analyze_h07_h08_endpoint_symmetry.py": {"git_blob": "1be562dd95eeba27fdca4682863c4bd8de2689fa", "sha256": "4ac4a1c88e995d8faae269971559cd1b62ba9aefa08dc972ec98f59787bbb4f0", "bytes": 40_892, "lines": 864},
    "workshops/corl2026_world_models/paper/data/h07_h08_endpoint_symmetry.json": {"git_blob": "b99248ca8663c4b4bc7335be7c4affca7090a566", "sha256": "cacaaad08f554e72a3f145618bc8c3c056286d85dae1ce53ba287b69eeeb4a51", "bytes": 87_382, "lines": 2_816},
    "workshops/corl2026_world_models/paper/data/h07_h08_endpoint_symmetry.csv": {"git_blob": "be84c3f6497a28f34cd93832dcc3a1e19e0ed5b0", "sha256": "d7d16b2b5a5c3c8d424438ac4d3e15766ce45a3aa5a1f3e1cfb89d24852faaac", "bytes": 17_314, "lines": 109},
    "workshops/corl2026_world_models/analysis/analyze_h09_h10_wording_control.py": {"git_blob": "42975b7a092a87f62e112b7bb80184ea5b1d4f5b", "sha256": "ddbb3a335d4c6247ef330ed3b0f9ce8b04d60e59cb2d7a5cf4f065289cb385fd", "bytes": 57_879, "lines": 1_071},
    "workshops/corl2026_world_models/paper/data/h09_h10_wording_control.json": {"git_blob": "30307c6b1061f1336bbcb34776520073fcf32c56", "sha256": "f1e7712dc86f3f4c70f463033016fdbe626009ee259754cc43609a24a1a98190", "bytes": 155_993, "lines": 3_611},
    "workshops/corl2026_world_models/paper/data/h09_h10_wording_control.csv": {"git_blob": "0501621dc62a227b96f800e6a7bf19c751e8ae0e", "sha256": "6541df85fe727438a7acdb11ee09f8bf2d6df5eb8ba7d3909f61a2f00a1fcf76", "bytes": 27_244, "lines": 121},
    "workshops/corl2026_world_models/analysis/analyze_h11_h12_phase_a_direct.py": {"git_blob": "9e5ecffa584be4aab8fe7dee495ce484adbc89b5", "sha256": "c61b914570cd9ff36a8e1f801b19a7d39109b66c7abc4786481ce06f493845fe", "bytes": 81_019, "lines": 1_422},
    "workshops/corl2026_world_models/paper/data/h11_h12_phase_a_direct.json": {"git_blob": "88f3e97f25c4f2f3c0530c6cbc8cf36107e178a6", "sha256": "7a942ab8d0987126e28c8ef933ad9ba12464df38f43ed7d4d5a6fed77414a610", "bytes": 91_511, "lines": 2_271},
    "workshops/corl2026_world_models/paper/data/h11_h12_phase_a_direct.csv": {"git_blob": "60b9d4f646b26bf93c406efc16491fade4d4b7b3", "sha256": "4f52da95a34a9f54758060f3f9587b87b9f1dccac49303d4ce94bc563c8e6b3f", "bytes": 14_885, "lines": 55},
    "workshops/corl2026_world_models/analysis/analyze_h13_v1_forecast_corpus.py": {"git_blob": "a901cfbfbcdcbd1f903fb9c7956c32206d3a74aa", "sha256": "a76a2308249ef721c35568ae725f60c84573d02fc0001c71abd6f85f90bf5027", "bytes": 35_020, "lines": 758},
    "workshops/corl2026_world_models/paper/data/h13_v1_forecast_corpus.json": {"git_blob": "238349a51d7044b548fa517650442392c5e3dc9a", "sha256": "fa2298bf6e10007a65b4eb922552aeb3dbde5006a413482ab31da325525c945a", "bytes": 16_910, "lines": 482},
    "workshops/corl2026_world_models/paper/data/h13_v1_forecast_corpus.csv": {"git_blob": "1e4f3ff3df62b839d9c5ca6d72e568ccfdfe10c1", "sha256": "389cf1ca1a826724a4b685e94043f0f7155fa1f125d271a8eb8ee941fb030fe7", "bytes": 726, "lines": 9},
    "workshops/corl2026_world_models/analysis/analyze_stopping_controls.py": {"git_blob": "514a6741e60335da9744784bd7cb554ce9f34ebf", "sha256": "e268522df8c50c49883acdc346ec16ee698f92a9ee683baa082b849cd217d77d", "bytes": 18_850, "lines": 308},
    "workshops/corl2026_world_models/results/stopping_controls.json": {"git_blob": "39d3e68dfdc1c66f0c87a6ebafa59b20964e7aaf", "sha256": "fc4660475fdc68d623fbea0f0572ae3249147c5512406a593c88d408c40ebcd5", "bytes": 259_226, "lines": 8_084},
    "workshops/corl2026_world_models/results/stopping_controls.csv": {"git_blob": "88346c4200ae609799756066a2cc4ff65b6f0990", "sha256": "72047c5261c9f60411689bdaafc54ff31fd5f21db738b4ef44412493fa8b0e75", "bytes": 47_644, "lines": 37},
    "workshops/corl2026_world_models/tests/test_stopping_controls.py": {"git_blob": "32440519b30ba6fa935892a818ad1586bc1e0009", "sha256": "42181c8e1f59274eb38861f6b242bd22b058cc39eb8c45a12e215740d6a4f550", "bytes": 12_399, "lines": 210},
    "workshops/corl2026_world_models/tests/test_h01_requested_region_entry.py": {"git_blob": "f7079df547ebead0b3b7d89b933d59631dd8753f", "sha256": "c7ae1dd0b657986c0bf91bfb8ef50aa5e53a3710c1db79ed726d61a7d96468da", "bytes": 12_234, "lines": 257},
    "workshops/corl2026_world_models/tests/test_h02_position_reflection.py": {"git_blob": "158df5d245355736e8dec8565587ee0b4f493748", "sha256": "49ef2c52b982fb0341cb96a6a3cbacfda1332cb048f350ec5a0ead4be21b167c", "bytes": 14_687, "lines": 326},
    "workshops/corl2026_world_models/tests/test_h03_h04_guidance_pilots.py": {"git_blob": "7f742057cb6911466ebfd39346d7dc6b0b8271b2", "sha256": "c3b85ce545c8a736953af7398d9194d4231a3d8cca1403b5158eee5455a1b9be", "bytes": 23_734, "lines": 475},
    "workshops/corl2026_world_models/tests/test_h05_lateral_dose_response.py": {"git_blob": "e2fb9522e3c0b1f36c2d49f8a717bbc8a5bbd2ab", "sha256": "13320b3e38564830c01bc37db389c20640aa3678a31e35e1ae7d60780e5134ba", "bytes": 11_474, "lines": 261},
    "workshops/corl2026_world_models/tests/test_h06_graded_symmetry.py": {"git_blob": "4c212298955c87240e6c3f4ae32f92c78516b6a7", "sha256": "f0328177cf660868cdd07ffc4b7ae67d2d42055b32dba9cc69904a90953918f0", "bytes": 8_134, "lines": 163},
    "workshops/corl2026_world_models/tests/test_h07_h08_endpoint_symmetry.py": {"git_blob": "673c8c239e1b73f5387f71305cac41e94159a113", "sha256": "28bec4f5399b1c88a06102187d08dee535871a5460e0288677bf8ef580b6db32", "bytes": 7_883, "lines": 166},
    "workshops/corl2026_world_models/tests/test_h09_h10_wording_control.py": {"git_blob": "298c6fe6408082ad2981697ded94b61363fabe43", "sha256": "72fd88adc274c99573a814105a1dfb92419b9fbe4c5efa2db0285d8561bb9e01", "bytes": 13_622, "lines": 254},
    "workshops/corl2026_world_models/tests/test_h11_h12_phase_a_direct.py": {"git_blob": "243cdbd9dd671c415701879fe7d549cb8ca4392f", "sha256": "07cf2db7bcd1e21f6a675fd956de1d95fc7b6a84954ccd08bb80906f4b06a6dc", "bytes": 21_538, "lines": 389},
    "workshops/corl2026_world_models/tests/test_h13_v1_forecast_corpus.py": {"git_blob": "c7832109fed5eadb34617b7bed119f9def79876a", "sha256": "660c56e228e2bd490994247d1aef63b68409f830b7afc694d64baf9a2b4bed7b", "bytes": 9_961, "lines": 201},
}

EXPECTED_NEW_ACTIVITY = {
    "annotation_media": 0,
    "behavioral_actions": 0,
    "behavioral_cells": 0,
    "confirmation_inference_cells": 0,
    "forecast_accuracy_estimates": 0,
    "human_responses": 0,
    "labels": 0,
    "model_requests": 0,
    "queue_descriptors": 0,
    "request_rows": 0,
    "request_selections": 0,
    "retrieval_jobs": 0,
    "runnable_manifests": 0,
}
EXPECTED_GATE_STATE = {
    "annotation_media_ready": False,
    "camera_alignment_ready": False,
    "confirmation_release_ready": False,
    "current_GM_PVC_worker_state_reconciled": False,
    "development_annotation_gate_ready": False,
    "human_annotation_ready": False,
    "live_quiescence_inferred": False,
    "resource_freeze_ready": False,
    "safe_to_release_new_work": False,
}

PACKAGE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "package_id": "h01_requested_region_entry_and_stopping",
        "historical_ids": ("H01",),
        "stem": "h01_requested_region_entry",
        "schema_version": "wmf-retained-h01-requested-region-entry-v1",
        "csv_rows": 27,
        "test_paths": (
            "workshops/corl2026_world_models/tests/test_stopping_controls.py",
            "workshops/corl2026_world_models/tests/test_h01_requested_region_entry.py",
        ),
        "build_markers": (
            'run("stopping-time controls", [sys.executable, str(package / "analysis/analyze_stopping_controls.py")])',
            'run("H01 requested-region entry", [sys.executable, str(package / "analysis/analyze_h01_requested_region_entry.py")])',
        ),
        "boundary_mode": "H01_explicit_claim_boundary_plus_stopping_scope",
        "false_fields": (
            "action_index_is_physical_time", "confirmation_test_or_stage_gate",
            "first_entry_is_behavioral_success", "forecast_accuracy_evidence",
            "matched_seed_labels_are_independent_scenes", "new_human_labels",
            "new_model_inference", "new_robot_episodes",
            "sustained_entry_has_common_post_entry_observation_window",
        ),
    },
    {
        "package_id": "h02_position_reflection",
        "historical_ids": ("H02",), "stem": "h02_position_reflection",
        "schema_version": "wmf-retained-h02-position-reflection-v1", "csv_rows": 27,
        "test_paths": ("workshops/corl2026_world_models/tests/test_h02_position_reflection.py",),
        "build_markers": ('run("H02 retained position reflection", [sys.executable, str(package / "analysis/analyze_h02_position_reflection.py")])',),
        "boundary_mode": "explicit_zero_activity_and_closed_gate_schema",
        "false_fields": (
            "advances_any_scientific_gate", "changes_232_cell_core_bound",
            "default_DreamZero_claim", "entry_action_index_is_physical_time",
            "forecast_accuracy_evidence", "independent_model_noise_replicates",
            "new_inference_or_episodes", "official_D1_evidence",
            "physical_time_evidence", "raw_trace_recovery",
            "requested_region_entry_is_registered_estimand",
        ),
    },
    {
        "package_id": "h03_h04_guidance_pilots",
        "historical_ids": ("H03", "H04"), "stem": "h03_h04_guidance_pilots",
        "schema_version": "gm-forecast-ablation-retained-h03-h04-guidance-pilots-v1", "csv_rows": 12,
        "test_paths": ("workshops/corl2026_world_models/tests/test_h03_h04_guidance_pilots.py",),
        "build_markers": ('run("H03 H04 retained guidance pilots", [sys.executable, str(package / "analysis/analyze_h03_h04_guidance_pilots.py")])',),
        "boundary_mode": "explicit_zero_activity_and_closed_gate_schema",
        "false_fields": (
            "changes_232_cell_core_bound", "confirmation_test_or_stage_gate",
            "forecast_accuracy_claim", "forecast_accuracy_evidence",
            "independent_scene_generalization", "models_pooled", "new_human_labels",
            "new_model_inference", "new_robot_episodes",
            "powered_or_general_performance_claim",
        ),
    },
    {
        "package_id": "h05_lateral_dose_response",
        "historical_ids": ("H05",), "stem": "h05_lateral_dose_response",
        "schema_version": "gm-forecast-ablation-retained-h05-dose-response-v1", "csv_rows": 105,
        "test_paths": ("workshops/corl2026_world_models/tests/test_h05_lateral_dose_response.py",),
        "build_markers": ('run("H05 retained lateral dose response", [sys.executable, str(package / "analysis/analyze_h05_lateral_dose_response.py")])',),
        "boundary_mode": "explicit_zero_activity_and_closed_gate_schema",
        "false_fields": (
            "changes_232_cell_core_bound", "confirmation_test_or_stage_gate",
            "controlled_positions_are_independent_scenes", "forecast_accuracy_evidence",
            "matched_seeds_are_independent_scenes", "new_human_labels",
            "new_model_inference", "new_robot_episodes",
        ),
    },
    {
        "package_id": "h06_graded_symmetry",
        "historical_ids": ("H06",), "stem": "h06_graded_symmetry",
        "schema_version": "wmf-retained-h06-graded-symmetry-v1", "csv_rows": 108,
        "test_paths": ("workshops/corl2026_world_models/tests/test_h06_graded_symmetry.py",),
        "build_markers": ('run("H06 retained graded symmetry", [sys.executable, str(package / "analysis/analyze_h06_graded_symmetry.py")])',),
        "boundary_mode": "explicit_zero_activity_and_closed_gate_schema",
        "false_fields": (
            "changes_232_cell_core_bound", "confirmation_test_or_stage_gate",
            "equivalence_claim", "forecast_accuracy_evidence", "new_human_labels",
            "new_model_inference", "new_robot_episodes",
            "position_only_reflection_replication", "primary_rows_are_independent_scenes",
        ),
    },
    {
        "package_id": "h07_h08_endpoint_symmetry",
        "historical_ids": ("H07", "H08"), "stem": "h07_h08_endpoint_symmetry",
        "schema_version": "wmf-retained-h07-h08-endpoint-symmetry-v1", "csv_rows": 108,
        "test_paths": ("workshops/corl2026_world_models/tests/test_h07_h08_endpoint_symmetry.py",),
        "build_markers": ('run("H07 H08 retained endpoint symmetry", [sys.executable, str(package / "analysis/analyze_h07_h08_endpoint_symmetry.py")])',),
        "boundary_mode": "explicit_zero_activity_and_closed_gate_schema",
        "false_fields": (
            "changes_232_cell_core_bound", "confirmation_test_or_stage_gate",
            "equivalence_claim", "forecast_accuracy_evidence", "new_human_labels",
            "new_model_inference", "new_robot_episodes",
            "position_only_reflection_replication", "rows_are_independent_scenes",
        ),
    },
    {
        "package_id": "h09_h10_wording_control",
        "historical_ids": ("H09", "H10"), "stem": "h09_h10_wording_control",
        "schema_version": "wmf-retained-h09-h10-wording-control-v1", "csv_rows": 120,
        "test_paths": ("workshops/corl2026_world_models/tests/test_h09_h10_wording_control.py",),
        "build_markers": ('run("H09 H10 retained wording control", [sys.executable, str(package / "analysis/analyze_h09_h10_wording_control.py")])',),
        "boundary_mode": "explicit_zero_activity_and_closed_gate_schema",
        "false_fields": (
            "changes_232_cell_core_bound", "confirmation_test_or_stage_gate",
            "confirmatory_wording_claim", "equivalence_claim", "forecast_accuracy_evidence",
            "models_pooled", "new_human_labels", "new_model_inference",
            "new_robot_episodes", "rows_are_independent_scenes",
        ),
    },
    {
        "package_id": "h11_h12_phase_a_direct",
        "historical_ids": ("H11", "H12"), "stem": "h11_h12_phase_a_direct",
        "schema_version": "wmf-retained-h11-h12-phase-a-direct-v1", "csv_rows": 54,
        "test_paths": ("workshops/corl2026_world_models/tests/test_h11_h12_phase_a_direct.py",),
        "build_markers": ('run("H11 H12 retained Phase A direct commands", [sys.executable, str(package / "analysis/analyze_h11_h12_phase_a_direct.py")])',),
        "boundary_mode": "explicit_zero_activity_and_closed_gate_schema",
        "false_fields": (
            "annotation_evidence", "changes_232_cell_core_bound",
            "confirmation_test_or_stage_gate", "equivalence_claim",
            "forecast_accuracy_evidence", "independent_scene_claim", "models_pooled",
            "new_model_inference", "new_robot_episodes",
        ),
    },
    {
        "package_id": "h13_v1_forecast_corpus",
        "historical_ids": ("H13",), "stem": "h13_v1_forecast_corpus",
        "schema_version": "h13-v1-retained-evaluator-package-v1", "csv_rows": 8,
        "test_paths": ("workshops/corl2026_world_models/tests/test_h13_v1_forecast_corpus.py",),
        "build_markers": ('run("H13 retained V1 evaluator corpus", [sys.executable, str(package / "analysis/analyze_h13_v1_forecast_corpus.py")])',),
        "boundary_mode": "explicit_zero_activity_and_closed_gate_schema",
        "false_fields": (
            "annotation_or_release_gate_advanced", "core_232_cell_accounting_changed",
            "forecast_fidelity_or_accuracy_estimated", "new_model_inference",
        ),
    },
)

SPEC_PATH = "workshops/corl2026_world_models/docs/ABLATION_SPEC.md"
INVENTORY_PATH = "workshops/corl2026_world_models/results/existing_ablation_inventory.json"
STOPPING_JSON = "workshops/corl2026_world_models/results/stopping_controls.json"
STOPPING_CSV = "workshops/corl2026_world_models/results/stopping_controls.csv"
EXPECTED_IDS = tuple(f"H{number:02d}" for number in range(1, 14))
PAPER_MARKERS = {
    "H01": "retained H01 A7-style decomposition",
    "H02": "Retained H02 compact audit",
    "H03": "H03/H04 post-result guidance pilots",
    "H04": "H03/H04 post-result guidance pilots",
    "H05": "H05 dose response",
    "H06": "H06: 0.020",
    "H07": "H07/H08 (DreamZero-s2/Edge)",
    "H08": "H07/H08 (DreamZero-s2/Edge)",
    "H09": "H09/H10 (Nano/Edge)",
    "H10": "H09/H10 (Nano/Edge)",
    "H11": "H11/H12 are separate fixed-scene Phase-A direct-command cohorts",
    "H12": "H12's totals match H07 numerically but use distinct seeds and are not a replication",
    "H13": "H13 evaluator-development audit",
}
ADDITIONAL_ROLES = {
    "H01": (
        "primary_behavioral_paper:nano_reflection",
        "stopping_sensitivity_supplement",
        "requested_region_entry_supplement",
        "representative_media:presentation_only",
    ),
    "H02": (
        "primary_behavioral_paper:dreamzero_reflection",
        "compact_position_reflection_audit",
    ),
    "H07": (
        "primary_behavioral_paper:dreamzero_symmetry",
        "endpoint_symmetry_audit",
    ),
}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{label}[{index}]")


def parse_json(raw: bytes, *, label: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {label}: {error}") from error
    _reject_nonfinite(value, label)
    return value


def parse_csv(raw: bytes, *, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 CSV in {label}") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames
    if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
        raise ValueError(f"invalid or duplicate CSV field in {label}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"malformed CSV row in {label}")
    return fields, rows


def authenticate_sources(root: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    commit = _git(root, "rev-parse", "--verify", BASE_COMMIT + "^{commit}").decode().strip()
    if commit != BASE_COMMIT:
        raise ValueError("base commit did not resolve exactly")
    tree = _git(root, "rev-parse", "--verify", BASE_COMMIT + "^{tree}").decode().strip()
    if tree != BASE_TREE:
        raise ValueError("base tree identity changed")
    raw_by_path: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    for path, expected in EXPECTED_SOURCES.items():
        object_id = _git(root, "rev-parse", "--verify", f"{BASE_COMMIT}:{path}").decode().strip()
        if object_id != expected["git_blob"]:
            raise ValueError(f"pinned Git blob changed: {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        actual = {
            "path": path, "git_blob": object_id, "sha256": _sha256(raw),
            "bytes": len(raw), "lines": len(raw.splitlines()),
        }
        if {key: actual[key] for key in expected} != expected:
            raise ValueError(f"pinned source bytes changed: {path}")
        checkout_path = root / path
        protected_result = path.startswith("workshops/corl2026_world_models/results/")
        # These integration outputs intentionally evolve after the pinned coverage
        # snapshot.  The audit still authenticates and parses their exact base Git
        # objects; it only skips current-checkout byte equality for this narrow set.
        evolving_integration = path in {PAPER_PATH, BUILD_PATH, PAPER_RESULTS_PATH}
        if checkout_path.is_file():
            if not evolving_integration and checkout_path.read_bytes() != raw:
                raise ValueError(f"checkout diverges from pinned source: {path}")
        elif not protected_result:
            raise ValueError(f"required checkout source is absent: {path}")
        raw_by_path[path] = raw
        receipts.append(actual)
    ordered = sorted(receipts, key=lambda item: item["path"])
    manifest = "".join(
        f"{item['path']}\0{item['git_blob']}\0{item['bytes']}\0{item['lines']}\0{item['sha256']}\n"
        for item in ordered
    ).encode("utf-8")
    aggregate = {
        "files": len(ordered),
        "bytes": sum(item["bytes"] for item in ordered),
        "lines": sum(item["lines"] for item in ordered),
        "manifest_sha256": _sha256(manifest),
    }
    if aggregate != EXPECTED_SOURCE_AGGREGATE:
        raise ValueError(f"authenticated-source aggregate changed: {aggregate}")
    return raw_by_path, ordered


def parse_spec_table(raw: bytes) -> list[dict[str, str]]:
    text = raw.decode("utf-8", errors="strict")
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("| H"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != 4:
            raise ValueError("registered historical table row no longer has four columns")
        rows.append(dict(zip(("id", "experiment", "design_and_result", "use_in_paper"), cells)))
    ids = tuple(row["id"] for row in rows)
    if ids != EXPECTED_IDS:
        raise ValueError(f"registered historical IDs changed: {ids}")
    required_text = (
        "Results are kept in their original cohorts; the table is not a pooled benchmark.",
        "No historical collection above is an unseen confirmation set for the new forecast metric.",
        "Raw H01/H02 files referenced under the old execution paths are absent on this machine.",
        "Their manifests are recovery pointers, not proof that the original host still has the files.",
        "Treat this as a retrospective sensitivity analysis until reproduced by a committed analysis script.",
        "validated physical-time pairing of generated and execution images",
        "independent image labels",
        "prospective varied-layout test of forecast accuracy",
        "The maximum core workload is **232 new episodes**.",
    )
    for statement in required_text:
        if statement not in text:
            raise ValueError(f"required registered boundary is absent: {statement}")
    return rows


def parse_inventory(raw: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = parse_json(raw, label=INVENTORY_PATH)
    if not isinstance(document, dict) or set(document) != {
        "checked_on", "entries", "media_status", "review_analysis", "scope",
        "upstream_pin", "version",
    }:
        raise ValueError("historical inventory top-level schema changed")
    if document["scope"] != "Relevant completed WAM cohorts; no pooling or sum across table entries":
        raise ValueError("historical inventory no-pooling boundary changed")
    entries = document["entries"]
    if not isinstance(entries, list) or tuple(entry.get("id") for entry in entries) != EXPECTED_IDS:
        raise ValueError("historical inventory IDs changed")
    for entry in entries:
        if entry.get("status") != "completed_historical_evidence":
            raise ValueError(f"historical status changed: {entry.get('id')}")
        if entry.get("confirmation_eligible") is not False:
            raise ValueError(f"historical cohort became confirmation eligible: {entry.get('id')}")
        if type(entry.get("episodes")) is not int or entry["episodes"] <= 0:
            raise ValueError(f"invalid per-cohort episode count: {entry.get('id')}")
        if not isinstance(entry.get("source_bindings"), list) or not entry["source_bindings"]:
            raise ValueError(f"missing source bindings: {entry.get('id')}")
    review = document["review_analysis"]
    if review != {
        "nano_pretermination_ordering": [27, 27],
        "status": "checked during review; standalone reproducible script not yet packaged",
    }:
        raise ValueError("pinned review-analysis snapshot changed")
    return document, entries


def _source_path(kind: str, stem: str) -> str:
    if kind == "analyzer":
        return f"workshops/corl2026_world_models/analysis/analyze_{stem}.py"
    return f"workshops/corl2026_world_models/paper/data/{stem}.{kind}"


def _require_false_fields(boundary: Mapping[str, Any], fields: Sequence[str], package_id: str) -> None:
    for field in fields:
        if boundary.get(field) is not False:
            raise ValueError(f"{package_id} boundary is not explicitly false: {field}")


def _receipt(path: str) -> dict[str, Any]:
    return {"path": path, **EXPECTED_SOURCES[path]}


def validate_package_integration(raw: Mapping[str, bytes]) -> dict[str, Any]:
    paper = raw[PAPER_PATH].decode("utf-8", errors="strict")
    build = raw[BUILD_PATH].decode("utf-8", errors="strict")
    manuscript: dict[str, dict[str, Any]] = {}
    for historical_id, marker in PAPER_MARKERS.items():
        if marker not in paper:
            raise ValueError(f"manuscript coverage marker absent for {historical_id}")
        manuscript[historical_id] = {"marker": marker, "covered": True}

    package_tests: dict[str, list[dict[str, Any]]] = {}
    build_markers: list[str] = []
    for package_spec in PACKAGE_SPECS:
        package_id = package_spec["package_id"]
        test_receipts: list[dict[str, Any]] = []
        for test_path in package_spec["test_paths"]:
            test_text = raw[test_path].decode("utf-8", errors="strict")
            if test_path.endswith("test_stopping_controls.py"):
                expected_analyzer = "analyze_stopping_controls.py"
            else:
                expected_analyzer = f"analyze_{package_spec['stem']}.py"
            if expected_analyzer not in test_text:
                raise ValueError(f"focused test is not bound to analyzer: {test_path}")
            test_receipts.append(_receipt(test_path))
        package_tests[package_id] = test_receipts
        for marker in package_spec["build_markers"]:
            if build.count(marker) != 1:
                raise ValueError(f"build-stage marker is not unique: {marker}")
            build_markers.append(marker)
    if [build.index(marker) for marker in build_markers] != sorted(
        build.index(marker) for marker in build_markers
    ):
        raise ValueError("retained package build stages changed order")

    primary_marker = (
        'run("paper experiment results", '
        '[sys.executable, str(package / "analysis/extract_paper_results.py")])'
    )
    media_marker = (
        'run("representative H01 media", '
        '[sys.executable, str(package / "analysis/package_representative_media.py")])'
    )
    for marker in (primary_marker, media_marker):
        if build.count(marker) != 1:
            raise ValueError(f"additional-role build marker changed: {marker}")
    paper_results_test = raw[
        "workshops/corl2026_world_models/tests/test_paper_results.py"
    ].decode("utf-8", errors="strict")
    if "extract_paper_results.py" not in paper_results_test:
        raise ValueError("primary-paper focused test is not bound to its extractor")
    media_test = raw[
        "workshops/corl2026_world_models/tests/test_representative_media_package.py"
    ].decode("utf-8", errors="strict")
    if "package_representative_media.py" not in media_test:
        raise ValueError("representative-media focused test is not bound to its packager")

    primary = parse_json(raw[PAPER_RESULTS_PATH], label=PAPER_RESULTS_PATH)
    if not isinstance(primary, dict) or primary.get("schema") != "completed-world-action-model-behavior-paper-v1":
        raise ValueError("primary behavioral-paper schema changed")
    if set(primary.get("cohorts", {})) != {
        "nano_reflection", "dreamzero_reflection", "dreamzero_symmetry",
    }:
        raise ValueError("primary behavioral-paper cohort roles changed")
    if primary.get("inference_scope") != (
        "Original within-cohort seed-block statistics; no pooled model, cohort, or arena "
        "inference. The fixed physical layouts do not establish generalization to unseen scenes."
    ):
        raise ValueError("primary behavioral-paper no-pooling boundary changed")

    media = parse_json(raw[PRESENTATION_MEDIA_PATH], label=PRESENTATION_MEDIA_PATH)
    if not isinstance(media, dict) or media.get("schema_version") != "wmf-h01-representative-media-package-v1":
        raise ValueError("representative-media schema changed")
    totals = media.get("totals", {})
    if (totals.get("new_model_requests"), totals.get("new_behavioral_episodes"), totals.get("new_human_labels")) != (0, 0, 0):
        raise ValueError("representative media records new scientific activity")
    media_boundary = media.get("claim_boundary", {})
    for field in (
        "not_scientific_timing_maps", "not_physical_time_alignment",
        "not_forecast_accuracy_evidence", "not_annotation_evidence",
        "does_not_complete_a5_or_any_forecast_gate",
    ):
        if media_boundary.get(field) is not True:
            raise ValueError(f"representative-media boundary changed: {field}")

    return {
        "coverage_snapshot": {"commit": BASE_COMMIT, "tree": BASE_TREE},
        "manuscript": {
            "source": _receipt(PAPER_PATH),
            "all_13_historical_ids_covered": True,
            "historical_id_markers": manuscript,
        },
        "build": {
            "source": _receipt(BUILD_PATH),
            "all_9_grouped_analysis_packages_executed": True,
            "H01_stopping_supplement_executed": True,
            "primary_behavioral_paper_extractor_executed": True,
            "representative_H01_media_packager_executed": True,
        },
        "focused_tests": package_tests,
        "primary_behavioral_paper": {
            "historical_id_roles": {
                "H01": "nano_reflection", "H02": "dreamzero_reflection",
                "H07": "dreamzero_symmetry",
            },
            "extractor": _receipt("workshops/corl2026_world_models/analysis/extract_paper_results.py"),
            "canonical_json": _receipt(PAPER_RESULTS_PATH),
            "focused_test": _receipt("workshops/corl2026_world_models/tests/test_paper_results.py"),
            "role_is_additional_view_not_new_cohort": True,
        },
        "representative_H01_media": {
            "role": "presentation_only_not_scientific_timing_or_outcome_coverage",
            "packager": _receipt("workshops/corl2026_world_models/analysis/package_representative_media.py"),
            "canonical_manifest": _receipt(PRESENTATION_MEDIA_PATH),
            "focused_test": _receipt("workshops/corl2026_world_models/tests/test_representative_media_package.py"),
            "role_is_additional_view_not_new_cohort": True,
        },
    }


def analyze(root: Path = ROOT) -> dict[str, Any]:
    raw, source_receipts = authenticate_sources(root)
    spec_rows = parse_spec_table(raw[SPEC_PATH])
    inventory, inventory_entries = parse_inventory(raw[INVENTORY_PATH])
    integration = validate_package_integration(raw)
    inventory_by_id = {entry["id"]: entry for entry in inventory_entries}
    spec_by_id = {row["id"]: row for row in spec_rows}

    packages: list[dict[str, Any]] = []
    package_by_historical_id: dict[str, dict[str, Any]] = {}
    for package_spec in PACKAGE_SPECS:
        stem = package_spec["stem"]
        analyzer_path = _source_path("analyzer", stem)
        json_path = _source_path("json", stem)
        csv_path = _source_path("csv", stem)
        document = parse_json(raw[json_path], label=json_path)
        if not isinstance(document, dict) or document.get("schema_version") != package_spec["schema_version"]:
            raise ValueError(f"canonical JSON schema version changed: {package_spec['package_id']}")
        _, csv_rows = parse_csv(raw[csv_path], label=csv_path)
        if len(csv_rows) != package_spec["csv_rows"]:
            raise ValueError(f"canonical CSV row count changed: {package_spec['package_id']}")
        boundary = document.get("claim_boundary")
        if not isinstance(boundary, dict):
            raise ValueError(f"claim boundary missing: {package_spec['package_id']}")
        _require_false_fields(boundary, package_spec["false_fields"], package_spec["package_id"])
        if package_spec["historical_ids"] == ("H01",):
            if "new_activity" in document or "gate_state" in document:
                raise ValueError("H01 boundary mode unexpectedly changed")
        else:
            if document.get("new_activity") != EXPECTED_NEW_ACTIVITY:
                raise ValueError(f"new-activity receipt changed: {package_spec['package_id']}")
            if document.get("gate_state") != EXPECTED_GATE_STATE:
                raise ValueError(f"gate-state receipt changed: {package_spec['package_id']}")
        package_record = {
            "package_id": package_spec["package_id"],
            "historical_ids": list(package_spec["historical_ids"]),
            "schema_version": package_spec["schema_version"],
            "analyzer": {"path": analyzer_path, **EXPECTED_SOURCES[analyzer_path]},
            "canonical_json": {"path": json_path, **EXPECTED_SOURCES[json_path]},
            "canonical_csv": {
                "path": csv_path, **EXPECTED_SOURCES[csv_path], "data_rows": len(csv_rows),
            },
            "boundary_mode": package_spec["boundary_mode"],
            "claim_boundary_closed": True,
            "focused_tests": integration["focused_tests"][package_spec["package_id"]],
            "base_build_stage_markers": list(package_spec["build_markers"]),
        }
        packages.append(package_record)
        for historical_id in package_spec["historical_ids"]:
            if historical_id in package_by_historical_id:
                raise ValueError(f"historical ID mapped more than once: {historical_id}")
            package_by_historical_id[historical_id] = package_record

    if tuple(sorted(package_by_historical_id)) != EXPECTED_IDS:
        raise ValueError("canonical package coverage is not exactly H01--H13")

    stopping = parse_json(raw[STOPPING_JSON], label=STOPPING_JSON)
    if not isinstance(stopping, dict) or stopping.get("schema_version") != "nano-h01-stopping-controls-v1":
        raise ValueError("H01 stopping-control schema changed")
    _, stopping_rows = parse_csv(raw[STOPPING_CSV], label=STOPPING_CSV)
    if len(stopping_rows) != 36:
        raise ValueError("H01 stopping-control CSV row count changed")
    status = stopping.get("definitions", {}).get("status")
    if status != "retrospective descriptive sensitivity analysis of historical H01; not prospective fixed-duration evidence":
        raise ValueError("H01 stopping-control scientific boundary changed")

    crosswalk: list[dict[str, Any]] = []
    for historical_id in EXPECTED_IDS:
        entry = inventory_by_id[historical_id]
        package_record = package_by_historical_id[historical_id]
        crosswalk.append({
            "historical_id": historical_id,
            "inventory_name": entry["name"],
            "inventory_study": entry["study"],
            "inventory_episode_count_not_summed": entry["episodes"],
            "spec_completed_experiment": spec_by_id[historical_id]["experiment"],
            "spec_use_in_paper": spec_by_id[historical_id]["use_in_paper"],
            "package_id": package_record["package_id"],
            "package_historical_ids": package_record["historical_ids"],
            "grouped_analysis_json_path": package_record["canonical_json"]["path"],
            "grouped_analysis_json_sha256": package_record["canonical_json"]["sha256"],
            "grouped_analysis_csv_path": package_record["canonical_csv"]["path"],
            "grouped_analysis_csv_sha256": package_record["canonical_csv"]["sha256"],
            "grouped_analysis_csv_data_rows": package_record["canonical_csv"]["data_rows"],
            "package_roles": list(ADDITIONAL_ROLES.get(historical_id, ("retained_grouped_analysis",))),
            "multiple_roles_are_not_additional_cohorts": True,
            "focused_test_paths": [item["path"] for item in package_record["focused_tests"]],
            "manuscript_marker": PAPER_MARKERS[historical_id],
            "confirmation_eligible": False,
            "claim_boundary_closed": True,
            "boundary_mode": package_record["boundary_mode"],
        })

    return {
        "schema_version": "wmf-retained-historical-evidence-coverage-audit-v1",
        "status": "all_registered_H01_H13_grouped_crosswalk_entries_authenticated_no_stage_advance",
        "provenance": {
            "base_commit": BASE_COMMIT,
            "base_tree": BASE_TREE,
            "authenticated_source_aggregate": EXPECTED_SOURCE_AGGREGATE,
            "authenticated_sources": source_receipts,
        },
        "registry": {
            "spec_path": SPEC_PATH,
            "inventory_path": INVENTORY_PATH,
            "upstream_pin": inventory["upstream_pin"],
            "registered_historical_ids": list(EXPECTED_IDS),
            "registered_historical_cohorts": 13,
            "grouped_analysis_packages": 9,
            "grouped_analysis_json_files": 9,
            "grouped_analysis_csv_files": 9,
            "H01_stopping_supplement": {
                "analyzer": {
                    "path": "workshops/corl2026_world_models/analysis/analyze_stopping_controls.py",
                    **EXPECTED_SOURCES["workshops/corl2026_world_models/analysis/analyze_stopping_controls.py"],
                },
                "json": {"path": STOPPING_JSON, **EXPECTED_SOURCES[STOPPING_JSON]},
                "csv": {"path": STOPPING_CSV, **EXPECTED_SOURCES[STOPPING_CSV], "data_rows": 36},
            },
            "all_registered_ids_mapped_exactly_once_to_grouped_analysis": True,
            "all_inventory_rows_remain_confirmation_ineligible": True,
            "episode_counts_summed_across_cohorts": False,
        },
        "packaging_completion": {
            "pinned_spec_and_inventory_review_status": "standalone reproducible script not yet packaged",
            "current_authenticated_H01_stopping_package": True,
            "interpretation": (
                "The pinned sentence records the earlier review state. The authenticated stopping analyzer "
                "and canonical outputs now complete that packaging task; this is not new H01 evidence."
            ),
            "historical_inventory_rewritten": False,
            "checked_in_build_report_used_as_current_coverage_evidence": False,
            "build_report_boundary": (
                "The checked-in generated build report predates the complete retained-package stages; "
                "the pinned build source and package artifacts, not that stale report, define this audit."
            ),
        },
        "integration_coverage": integration,
        "packages": packages,
        "cohort_crosswalk": crosswalk,
        "coverage_summary": {
            "registered_cohorts": 13,
            "mapped_cohorts": 13,
            "unmapped_registered_cohorts": 0,
            "duplicate_mappings": 0,
            "grouped_analysis_packages": 9,
            "additional_primary_or_presentation_roles_counted_as_new_cohorts": 0,
            "packages_with_explicit_zero_activity_and_closed_gate_schema": 8,
            "H01_packages_with_equivalent_explicit_claim_boundary": 1,
            "packages_with_closed_claim_boundaries": 9,
            "packaging_coverage_complete": True,
            "scientific_confirmation_claims_created_by_this_audit": 0,
        },
        "historical_custom_s2_ids_not_official_D1": ["H02", "H04", "H07", "H12"],
        "preserved_evidence_gaps": {
            "raw_H01_H02_current_local_availability_established": False,
            "recovery_manifest_proves_original_host_availability": False,
            "validated_physical_time_pairing": False,
            "independent_image_labels": False,
            "qualified_default_or_conditional_DreamZero_predictions": False,
            "prospective_varied_layout_forecast_accuracy": False,
            "prospective_A3_A9_completed": False,
            "fixed_duration_H01_control_completed_by_retrospective_stopping_analysis": False,
        },
        "claim_boundary": {
            "allowed_interpretation": (
                "All 13 registered historical cohort IDs map exactly once to an authenticated grouped "
                "analysis package and explicit closed claim boundary; additional primary, supplemental, "
                "and presentation roles are recorded without creating more cohorts."
            ),
            "packaging_coverage_is_scientific_confirmation": False,
            "scientific_outcomes_reestimated": False,
            "models_or_cohorts_pooled": False,
            "episode_counts_summed": False,
            "equivalence_established": False,
            "independent_scene_or_layout_generalization_established": False,
            "language_understanding_established": False,
            "forecast_accuracy_or_fidelity_estimated": False,
            "raw_or_replay_evidence_recovered": False,
            "new_scientific_evidence_created": False,
            "current_worker_or_queue_state_inferred": False,
            "changes_232_cell_core_bound": False,
            "maximum_core_behavioral_cells": 232,
        },
        "new_activity": EXPECTED_NEW_ACTIVITY,
        "gate_state": EXPECTED_GATE_STATE,
    }


CSV_FIELDS = (
    "historical_id", "inventory_name", "inventory_study",
    "inventory_episode_count_not_summed", "package_id", "package_historical_ids",
    "grouped_analysis_json_path", "grouped_analysis_json_sha256",
    "grouped_analysis_csv_path", "grouped_analysis_csv_sha256",
    "grouped_analysis_csv_data_rows", "package_roles",
    "multiple_roles_are_not_additional_cohorts", "focused_test_paths",
    "manuscript_marker", "confirmation_eligible", "claim_boundary_closed",
    "boundary_mode",
)


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in report["cohort_crosswalk"]:
        rendered = {field: row[field] for field in CSV_FIELDS}
        rendered["package_historical_ids"] = ";".join(row["package_historical_ids"])
        rendered["package_roles"] = ";".join(row["package_roles"])
        rendered["focused_test_paths"] = ";".join(row["focused_test_paths"])
        writer.writerow(rendered)
    return stream.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = analyze(ROOT)
    rendered_json = render_json(report)
    rendered_csv = render_csv(report)
    if args.check:
        if not args.json.is_file() or args.json.read_text(encoding="utf-8") != rendered_json:
            raise ValueError(f"canonical JSON is absent or stale: {args.json}")
        if not args.csv.is_file() or args.csv.read_text(encoding="utf-8") != rendered_csv:
            raise ValueError(f"canonical CSV is absent or stale: {args.csv}")
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered_json, encoding="utf-8")
        args.csv.write_text(rendered_csv, encoding="utf-8")
    print(rendered_json, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
