# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import numpy as np

from calimu.imu.com_imu import ComImu
import serial


# stfu pycharm: I raised errors instead of using @abc.abstractmethod so that I could implement only the needed methods.
# noinspection PyAbstractClass
class MC6470IMU(ComImu):
    def set_accelerometer_offsets(self, mat4x4: np.ndarray, avg_scale: float):
        if self.connection is None or (not self.connection.is_open):
            raise RuntimeError("IMU should be connected to set offsets.")

        # debug: show old data
        self.connection.write(b"Q")
        self.connection.write(b"N")

        xyz_off = -mat4x4[3, 0:3]
        byte_msg = xyz_off.astype(np.float32).tobytes()
        self.connection.write(b"M")
        self.connection.write(byte_msg)

        xyz_mat = mat4x4[0:3, 0:3]
        xyz_mat = xyz_mat / avg_scale
        byte_msg = xyz_mat.astype(np.float32).tobytes()
        self.connection.write(b"P")
        self.connection.write(byte_msg)

    def set_magnetometer_offsets(self, mat4x4: np.ndarray, avg_scale: float):
        if self.connection is None or (not self.connection.is_open):
            raise RuntimeError("IMU should be connected to set offsets.")

        # debug: show old data
        self.connection.write(b"K")
        self.connection.write(b"H")

        xyz_off = -mat4x4[0:3, 3]
        byte_msg = xyz_off.astype(np.float32).tobytes()
        self.connection.write(b"G")
        self.connection.write(byte_msg)

        xyz_mat = mat4x4[0:3, 0:3]
        xyz_mat = xyz_mat / avg_scale
        xyz_mat = np.linalg.inv(xyz_mat)
        byte_msg = xyz_mat.astype(np.float32).tobytes()
        self.connection.write(b"J")
        self.connection.write(byte_msg)

    def mag_accel_iter(self):
        if self.connection is None or (not self.connection.is_open):
            return
        self.connection.write(b"a")  # mag loop
        self.connection.write(b"c")  # acc loop

        try:
            while True:
                line = self.connection.readline()
                print(line)
                try:
                    parts = line.split(b"[")
                    x = parts[1][0:-3]
                    y = parts[2][0:-3]
                    z = parts[3][0:-3]
                except IndexError as ie:
                    print(ie)  # transmission error, line garbled
                    continue
                try:
                    if b"Got mag data: " in line:
                        yield "m", int(x), int(y), int(z)
                    elif b"Got acc data: " in line:
                        yield "a", int(x), int(y), int(z)
                except ValueError as ve:
                    print(ve)  # another screw up: number was empty
                    continue
        finally:
            try:
                self.connection.write(b"b")  # stop mag loop
                self.connection.write(b"d")  # stop acc loop
            except serial.SerialException:
                pass  # already disconnected it seems

    def orientation_iter(self):
        self.connection.write(b"C")  # turn off acc display
        self.connection.write(b"D")  # turn off mag display
        self.connection.write(b"a")  # mag loop
        self.connection.write(b"c")  # acc loop
        self.connection.write(b"e")  # orient loop

        orient = np.eye(3)
        i = 0
        try:
            while True:
                line = self.connection.readline()
                try:
                    if line == b"Got orient:\r\n":
                        orient = np.eye(3)
                        i = 0
                    else:
                        line = line[1:-2]  # trim off start \t and end \r\n
                        nums = line.split(b", ")
                        orient[i, :] = [int(n) for n in nums]
                        i = (i + 1) % 3
                        if i == 0:  # looped back around. So it was fully filled.
                            yield orient
                except ValueError as ve:
                    print(ve)
                except IndexError as ie:
                    print(ie)

        finally:
            self.connection.write(b"b")  # stop mag loop
            self.connection.write(b"d")  # stop acc loop
            self.connection.write(b"f")  # stop orient loop
            self.connection.write(b"E")  # turn on acc display
            self.connection.write(b"F")  # turn on mag display
