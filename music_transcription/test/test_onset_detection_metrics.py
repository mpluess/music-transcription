import unittest
from unittest import TestCase

from music_transcription.onset_detection.metrics import onset_metric_times


class TestOnsetDetectionMetrics(TestCase):
    def test_onset_metric_times(self):
        onset_times = [0.1, 0.5, 1.3, 1.9]
        onset_times_predicted = [0.0799999998, 0.5200000005, 1.33, 1.5, 1.93]

        metrics_plus_minus_2 = onset_metric_times(onset_times, onset_times_predicted, 0.02)
        self.assertEqual(metrics_plus_minus_2.tp, 2)
        self.assertEqual(metrics_plus_minus_2.fn, 2)
        self.assertEqual(metrics_plus_minus_2.fp, 3)
        metrics_plus_minus_5 = onset_metric_times(onset_times, onset_times_predicted, 0.05)
        self.assertEqual(metrics_plus_minus_5.tp, 4)
        self.assertEqual(metrics_plus_minus_5.fn, 0)
        self.assertEqual(metrics_plus_minus_5.fp, 1)

if __name__ == '__main__':
    unittest.main()
