# -----------------------Tom Keane CID: 01788365----------------------
import unittest
import tfl_data
from datetime import datetime
from datetime import timedelta


class TestLineSeverity(unittest.TestCase):
    def test_base_run(self):
        tfl_data.api_log = []
        line_id = "circle"
        status = tfl_data.get_line_severity(line_id)
        self.assertTrue(type(status) == str)

    def test_addition_to_log(self):
        tfl_data.api_log = []
        line_id = "circle"
        tfl_data.get_line_severity(line_id)
        self.assertTrue(len(tfl_data.api_log) == 1)

    def test_correct_time_recording(self):
        tfl_data.api_log = []
        line_id = "circle"
        tfl_data.get_line_severity(line_id)
        self.assertTrue(datetime.now() - tfl_data.api_log[-1] < timedelta(seconds=1))

    def test_quota_reached(self):
        tfl_data.api_log = []
        line_id = "circle"
        tfl_data.api_log = [datetime.now() - timedelta(seconds=i) for i in range(5)]
        self.assertRaises(tfl_data.QuotaError, tfl_data.get_line_severity, line_id)

    def test_quota_refresh(self):
        tfl_data.api_log = []
        tfl_data.api_log = [datetime.now() - timedelta(minutes=5, seconds=i) for i in range(1, 5)]
        line_id = "circle"
        status = tfl_data.get_line_severity(line_id)
        self.assertTrue(type(status) == str)


"""
    def test_invalid_input_type(self):
        line_id = ["foo"]
        self.assertRaises(TypeError, tfl_data.get_line_severity, line_id)

    def test_invalid_input(self):
        line_id = "foo"
        self.assertRaises(ValueError, tfl_data.get_line_severity, line_id)

    def test_empty_str_input(self):
        line_id = ""
        self.assertRaises(ValueError, tfl_data.get_line_severity, line_id)
"""


class TestAirQuality(unittest.TestCase):

    def test_input_t(self):
        tfl_data.api_log = []
        is_future = True
        air_quality = tfl_data.get_air_quality(is_future)
        self.assertTrue(type(air_quality) == str)

    def test_input_f(self):
        tfl_data.api_log = []
        is_future = False
        air_quality = tfl_data.get_air_quality(is_future)
        self.assertTrue(type(air_quality) == str)

    def test_addition_to_log(self):
        tfl_data.api_log = []
        line_id = "circle"
        tfl_data.get_line_severity(line_id)
        self.assertTrue(len(tfl_data.api_log) == 1)

    def test_correct_time_recording(self):
        tfl_data.api_log = []
        is_future = False
        tfl_data.get_air_quality(is_future)
        self.assertTrue(datetime.now() - tfl_data.api_log[-1] < timedelta(seconds=1))

    def test_quota_reached(self):
        tfl_data.api_log = []
        tfl_data.api_log = [datetime.now() - timedelta(seconds=i) for i in range(5)]
        self.assertRaises(tfl_data.QuotaError, tfl_data.get_air_quality, False)

    def test_quota_refresh(self):
        tfl_data.api_log = []
        tfl_data.api_log = [datetime.now() - timedelta(minutes=5, seconds=i) for i in range(1, 5)]
        air_quality = tfl_data.get_air_quality(False)
        self.assertTrue(type(air_quality) == str)


"""
    def test_invalid_input_type(self):
        is_future = ["foo"]
        self.assertRaises(TypeError, tfl_data.get_air_quality, is_future)

    def test_invalid_input(self):
        is_future = None
        self.assertRaises(TypeError, tfl_data.get_air_quality, is_future)

    def test_empty_str_input(self):
        is_future = ""
        self.assertRaises(ValueError, tfl_data.get_air_quality, is_future)
"""


if __name__ == '__main__':
    unittest.main()

