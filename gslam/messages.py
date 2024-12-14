from enum import StrEnum, auto


class FrontendMessage(StrEnum):
    REQUEST_INITIALIZE = auto()
    ADD_KEYFRAME = auto()
    REQUEST_SYNC = auto()


class BackendMessage(StrEnum):
    SIGNAL_INITIALIZED = auto()
    SYNC = auto()
