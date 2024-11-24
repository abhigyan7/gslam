from enum import StrEnum, auto

class TrackerMessage(StrEnum):
    REQUEST_INITIALIZE = auto()
    REQUEST_RESET = auto()
    REQUEST_STOP = auto()
    REQUEST_SYNC = auto()


class MapMessage(StrEnum):
    ACKNOWLEDGE_STOP = auto()
    REQUEST_RESET = auto()
    REQUEST_STOP = auto()
    REQUEST_SYNC = auto()