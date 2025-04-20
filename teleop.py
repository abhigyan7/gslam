#!/usr/bin/env python3

import serial
import struct
from gslam.crc8 import crc8
import sys
import select

import time

START_BYTE = 0xA5

RED_TTL = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0'
# BLACK_TTL = '/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0'
BLACK_TTL = '/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5XK3RJT-if00-port0'
USING_TTL = BLACK_TTL


def calc_crc(data):
    hash_func = crc8()
    hash_func.update(data)
    return hash_func.digest()[0]


W = 0.4


class SerialNode:
    def __init__(self):
        self.serial_port = serial.Serial(USING_TTL, 115200)
        self.v = 0
        self.w = 0
        return

    def send(self, vw):
        # we don't need odom
        self.serial_port.reset_input_buffer()
        v, w = vw
        self.v = W * self.v + (1 - W) * v
        self.w = W * self.w + (1 - W) * w
        print(f'sending {(self.v,self.w)=}')
        data = [
            bytes(struct.pack("B", START_BYTE)),
            bytes(struct.pack("f", float(self.v))),
            bytes(struct.pack("f", float(self.w))),
        ]
        data = b''.join(data)
        hash = calc_crc(data[1:])
        data = [data, bytes(struct.pack('B', hash))]
        data = b''.join(data)
        self.serial_port.write(data)
        # print(f'wrote {data=} to serial')


def main(args=None):
    myserial = SerialNode()
    F = 0.25
    T = 0.9
    command_to_send = (0, 0)
    while True:
        try:
            i, _, _ = select.select([sys.stdin], [], [], 0.01)
            if i:
                command = sys.stdin.readline().strip()
                match command:
                    case 'w':
                        command_to_send = (F, 0)
                    case 's':
                        command_to_send = (-F, 0)
                    case 'a':
                        command_to_send = (0, T)
                    case 'd':
                        command_to_send = (0, -T)
                    case 'q':
                        command_to_send = (0, 0)
                        break
                    case _:
                        command_to_send = (0, 0)
            myserial.send(command_to_send)
        except KeyboardInterrupt:
            break
        time.sleep(0.15)
    command_to_send = (0, 0)
    myserial.serial_port.close()
    myserial.send(command_to_send)


if __name__ == '__main__':
    main()
