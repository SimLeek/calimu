# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import threading
import time
from threading import Lock

import numpy as np

from calimu.imu.devices.mc6470 import MC6470IMU
from calimu.imu.util import StoppableThread
import serial


class IMUPointStore(StoppableThread):
    def __init__(self, ardu, colors=None):
        super().__init__()

        self.ardu: MC6470IMU = ardu

        self.points = []
        self.point_colors = []

        self.mag_points = []
        self.acc_points = []
        self.latest_mag = [0, 0, -1]
        self.latest_acc = [0, -1, 0]

        self.lock = Lock()
        self.on = True

        self.lock_fps = 60

        self._gather_stop = threading.Event()
        self._gather_stop.set()

        self._display_orient_stop = threading.Event()
        self._display_orient_stop.set()

        if colors is None:
            self.colors = {
                "m": [0.8 * 255, 0.8 * 255, 0],
                "a": [1.0 * 255, 0, 0],
                "g": [0, 1.0 * 255, 0],
            }
        else:
            self.colors = colors

    def stop_gathering(self):
        self._gather_stop.set()

    def start_gathering(self):
        self._gather_stop.clear()

    def stop_displaying_orient(self):
        self._display_orient_stop.set()

    def start_displaying_orient(self):
        self._display_orient_stop.clear()

    def stop(self):
        # need to call ALL stop functions in order for the run loop to quit
        super().stop()
        self.stop_gathering()

    def __display_orient_loop(self, t0):
        for t, x, y, z in self.ardu.mag_accel_iter():
            self.lock.acquire()
            if t == "m":
                self.latest_mag = [x, y, z]
            elif t == "a":
                self.latest_acc = [x, y, z]
            self.lock.release()
            if time.time() - t0 > 1.0 / self.lock_fps:
                time.sleep(0)
                t0 = time.time()
            if self._display_orient_stop.isSet():
                break

    def __gather_loop(self, t0):
        for t, x, y, z in self.ardu.mag_accel_iter():
            self.lock.acquire()
            if t == "m":
                self.mag_points.append([x, y, z])
                self.latest_mag = [x, y, z]
            elif t == "a":
                self.acc_points.append([x, y, z])
                self.latest_acc = [x, y, z]
            self.points.append(np.asarray([x, y, z]))
            self.point_colors.append(self.colors[t])
            self.lock.release()
            if time.time() - t0 > 1.0 / self.lock_fps:
                time.sleep(0)
                t0 = time.time()
            if self._gather_stop.isSet():
                break

    def __default_loop(self):
        # by default, read and display everything from the arduino for debug purposes
        try:
            line = self.ardu.connection.readline()
            if line:
                print(line)
        except serial.SerialException:
            pass  # you somehow managed to disconnect inside the readline. Impressive.
        except TypeError:
            pass  # another read error
        except AttributeError:
            pass  # yet another read error

    def run(self):
        t0 = time.time()
        while not self.stopped():
            if not self._gather_stop.isSet():
                self.__gather_loop(t0)
            elif not self._display_orient_stop.isSet():
                self.__display_orient_loop(t0)
            elif self.ardu.connection is not None and self.ardu.connection.is_open:
                self.__default_loop()
