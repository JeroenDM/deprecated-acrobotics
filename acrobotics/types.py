from enum import Enum


class SamplingType(Enum):
    GRID = 0
    INCREMENTAL = 1
    MIN_INCREMENTAL = 2


class SampleMethod(Enum):
    random_uniform = 0
    deterministic_uniform = 1