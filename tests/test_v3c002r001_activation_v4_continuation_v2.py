"""Import-time ownership regression for the A004 continuation v2 runner."""

import importlib
import sys
import unittest


class ContinuationV2Tests(unittest.TestCase):
    def test_no_retry_runner_side_effects(self):
        retry_name = "experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_retry_runner"
        sys.modules.pop(retry_name, None)
        module = importlib.import_module(
            "experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_continuation_runner_v2"
        )
        base = importlib.import_module("experiments.v3.phase_c_semantic_equivalence_v3c002.runner")
        r001 = importlib.import_module("experiments.v3.phase_c_semantic_equivalence_v3c002r001.runner")
        self.assertNotIn(retry_name, sys.modules)
        self.assertEqual(base._next_attempt_root.__module__, base.__name__)
        self.assertEqual(base._dispatch_block.__module__, r001.__name__)
        self.assertIs(base.grouped_shard, module._remaining)


if __name__ == "__main__":
    unittest.main()
