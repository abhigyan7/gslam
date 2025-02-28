from enum import StrEnum, auto


class InitializationType(StrEnum):
    STRONG_INIT = auto()
    WEAK_INIT = auto()


class FrontendMessage(StrEnum):
    ADD_FRAME = auto()
    ADD_REFINED_DEPTHMAP = auto()
    REQUEST_INIT = auto()


class BackendMessage(StrEnum):
    SYNC = auto()
    END_SYNC = auto()
