# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import math

import numpy as np


class IMU(object):
    def get_accelerometer_scale(self):
        """Returns a number representing the max number of gs the accelerometer will detect."""
        raise NotImplementedError()

    def get_accelerometer_bits(self):
        """Returns an integer representing the number of bits the accelerometer uses."""
        raise NotImplementedError()

    def get_expected_accelerometer_1g(self):
        """Gets the number the accelerometer should report for 1g if it's perfectly calibrated."""
        div = math.log2(self.get_accelerometer_scale())
        expected = 2 ** int(self.get_accelerometer_bits() - div) - 1
        return expected

    def set_accelerometer_offsets(self, mat4x4: np.ndarray, avg_scale: float):
        """Sets the offset matrix for the accelerometer. avg_scale may be used to keep the default size."""
        raise NotImplementedError()

    def set_magnetometer_offsets(self, mat4x4: np.ndarray, avg_scale: float):
        """Sets the offset matrix for the magnetometer. avg_scale may be used to keep the default size."""
        raise NotImplementedError()

    # only one of mag, accel, or mag_accel_iter needs to be initialized
    def mag_iter(self):
        """Yield a for-loop capable iter to continuously get magnetometer data."""
        raise NotImplementedError()

    def accel_iter(self):
        """Yield a for-loop capable iter to continuously get accelerometer data."""
        raise NotImplementedError()

    def mag_accel_iter(self):
        """Yield a for-loop capable iter to continuously get all sensor data.
        This function can deal with combining setup and teardown."""
        raise NotImplementedError()

    def orientation_iter(self):
        """Yield a for-loop capable iter to continuously get orientation data.
        This may require sending different commands to the chip so that it can calculate orientation."""
        raise NotImplementedError()
