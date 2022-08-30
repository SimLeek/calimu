# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import time

from svtk.vtk_classes.vtk_animation_timer_callback import VTKAnimationTimerCallback
from svtk.vtk_classes.vtk_displayer import VTKDisplayer

from calimu.imu.store import IMUPointStore
import numpy as np

from calimu.imu.util import normalize


class _VTKIMUPointDisplayer(VTKAnimationTimerCallback):
    def __init__(self, point_store: IMUPointStore):
        super().__init__()
        self.point_store = point_store
        self.__is_first_loop = True

    def first_loop(self):
        self.add_points(
            [[0, 0, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0]],
            [[0, 0, 0], [0.4 * 255, 0.4 * 255, 0], [0.5 * 255, 0, 0], [0.6 * 255, 0.2 * 255, 0]]
        )

        self.add_lines(
            [2, 0, 1, 2, 0, 2, 2, 0, 3],
            [[0.4 * 255, 0.4 * 255, 0], [0.5 * 255, 0, 0], [0.6 * 255, 0.2 * 255, 0]]
        )

    def loop(self, obj, event):
        if self.__is_first_loop:
            self.first_loop()
            self.__is_first_loop = False

        if not self.point_store.lock.locked():
            self.point_store.lock.acquire()
            super(_VTKIMUPointDisplayer, self).loop(obj, event)
            if len(self.point_store.points) and len(
                    self.point_store.point_colors
            ) == len(self.point_store.points):
                self.add_points(
                    self.point_store.points,
                    self.point_store.point_colors,
                )



                self.fit_points_in_cam()
                self.point_store.points.clear()
                self.point_store.point_colors.clear()

            elen = np.linalg.norm(self.point_store.latest_mag)
            east = np.cross(normalize(self.point_store.latest_mag), -normalize(self.point_store.latest_acc))
            east = east * elen
            self.position_points(
                [self.point_store.latest_mag, self.point_store.latest_acc, east],
                [1, 2, 3]
            )

            self.point_store.lock.release()

            time.sleep(0)


class IMUPointDisplayer(object):
    def __init__(self, point_store: IMUPointStore, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.point_store = point_store
        self.displayer = VTKDisplayer(_VTKIMUPointDisplayer, self.point_store)

    def run_once(self) -> None:
        self.displayer.process_events()
