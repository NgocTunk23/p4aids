"""Checks for temporal boundaries and short target alignment."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from .chunked_data import ChunkedSequenceDataset
from .config import CONFIG


class ChunkedSequenceDatasetTest(unittest.TestCase):
    def test_gap_resets_context_and_short_target_matches_model_output(self):
        config = replace(CONFIG, data=replace(CONFIG.data, context_length=3,
                                               target_length=2, sequence_length=5, chunk_stride=2))
        frame = pd.DataFrame({
            "building_id": [1] * 5,
            "timestamp": pd.to_datetime([
                "2016-01-01 00:00", "2016-01-01 01:00", "2016-01-01 02:00",
                "2016-01-01 05:00", "2016-01-01 06:00",
            ]),
        })
        features = np.arange(10, 15, dtype=np.float32).reshape(-1, 1)
        labels = np.array([0, 1, 0, 1, 0], dtype=np.float32)
        dataset = ChunkedSequenceDataset(features, frame, np.arange(5), labels, config)

        self.assertEqual(len(dataset), 3)
        x, y, mask, pos, _ = dataset[1]
        np.testing.assert_array_equal(x[:, 0], [0, 0, 10, 11, 12])
        np.testing.assert_array_equal(y, [0, 0])
        np.testing.assert_array_equal(mask, [0, 1])
        np.testing.assert_array_equal(pos, [-1, 2])

        x_after_gap, _, mask_after_gap, pos_after_gap, _ = dataset[2]
        np.testing.assert_array_equal(x_after_gap[:, 0], [0, 0, 0, 13, 14])
        np.testing.assert_array_equal(mask_after_gap, [1, 1])
        np.testing.assert_array_equal(pos_after_gap, [3, 4])


if __name__ == "__main__":
    unittest.main()
