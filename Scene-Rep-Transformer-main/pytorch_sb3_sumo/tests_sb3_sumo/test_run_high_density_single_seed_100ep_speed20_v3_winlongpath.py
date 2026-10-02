from __future__ import annotations

import os
import unittest
from pathlib import Path

from tools import run_high_density_single_seed_100ep_speed20_v3_winlongpath as target


@unittest.skipUnless(os.name == "nt", "Windows extended paths are Windows-only")
class Speed20V3WindowsLongPathRunnerTest(unittest.TestCase):
    def test_default_result_root_is_extended_without_changing_subcommand(self) -> None:
        values = target._with_extended_result_root(["run", "--workers", "1"])

        self.assertEqual(values[0], "run")
        index = values.index("--result-root") + 1
        self.assertTrue(values[index].startswith(target.WINDOWS_EXTENDED_PREFIX))
        self.assertTrue(values[index].endswith("results_hd_ss100_s20_v3"))

    def test_existing_extended_result_root_is_idempotent(self) -> None:
        existing = Path(
            target.WINDOWS_EXTENDED_PREFIX
            + r"D:\speed20_v3\formal_results"
        )

        self.assertEqual(target.windows_extended_path(existing), existing)


if __name__ == "__main__":
    unittest.main()
