# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import copy
import time

import serial as pyserial
import serial.tools
import serial.tools.list_ports

from calimu.imu.imu import IMU


def list_ports():
    ports = serial.tools.list_ports.comports()
    # https://stackoverflow.com/a/52809180
    s = []
    for port, desc, hwid in sorted(ports):
        s.append("{}: {} [{}]".format(port, desc, hwid))
    return s


# stfu pycharm: this subclass of an abstract class is also meant to be an abstract class
# noinspection PyAbstractClass
class ComImu(IMU):
    DEFAULT_BAUDRATE = 250000  # IDK a good reason not to

    # this setup works for Arduino. Untested with other stuff.
    DEFAULT_PORT_CONFIG = {
        "baudrate": DEFAULT_BAUDRATE,
        "bytesize": pyserial.EIGHTBITS,
        "parity": pyserial.PARITY_NONE,
        "stopbits": pyserial.STOPBITS_ONE,
        "timeout": 0.05,  # read timeout
        "xonxoff": False,
        "rtscts": False,
    }

    def __init__(self, port=None, baud=DEFAULT_BAUDRATE, **kwargs):
        self.connection = None
        if port is not None:
            self.connection = self._get_serial(port, baud, **kwargs)
        self.end_writes = []

    @staticmethod
    def _get_serial(port=None, baud=DEFAULT_BAUDRATE, **kwargs) -> pyserial.Serial:
        if port is None:
            s = ["\t", list_ports()]
            s_all = "\n\t".join(s)
            raise ValueError("Please specify port. Available ports:\n" + s_all)

        if isinstance(port, str):
            port_config = copy.deepcopy(ComImu.DEFAULT_PORT_CONFIG)
            port_config["baudrate"] = baud
            port_config.update(**kwargs)
            proc = pyserial.serial_for_url(port, **port_config)
        else:  # pyserial instance
            proc = port
            proc.timeout = ComImu.DEFAULT_PORT_CONFIG["timeout"]  # set read timeout

        time.sleep(2)  # wait for arduino to start up
        return proc

    def connect(self, port, baud=DEFAULT_BAUDRATE, **kwargs):
        self.connection = self._get_serial(port, baud, **kwargs)

    def disconnect(self):
        self.connection.close()
        self.connection = None
