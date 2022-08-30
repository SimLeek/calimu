# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import threading
from threading import Thread

import numpy as np


class StoppableThread(Thread):
    # https://stackoverflow.com/a/325528
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()  # renamed due to new internal stuff

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.isSet()


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
