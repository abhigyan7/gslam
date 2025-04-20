#!/usr/bin/env python3

from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080P', fps=15, encode='H265')

    # Synchronize & save all (encoded) streams
    oak.record([color.out.encoded], './', RecordType.VIDEO)
    # Show color stream
    # oak.visualize([color.out.camera], scale=2/3, fps=True)

    oak.start(blocking=True)
