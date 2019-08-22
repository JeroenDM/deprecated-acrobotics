from typing import List
from enum import Enum
from .samplers import SampleMethod
from .geometry import Scene
from typing import List
from .path.path_pt import PathPt


class SamplingType(Enum):
    GRID = 0
    INCREMENTAL = 1
    MIN_INCREMENTAL = 2


class PlanningSetting:
    def __init__(
        self,
        sampling_type: SamplingType,
        sample_method: SampleMethod = None,
        num_samples: int = None,
        desired_num_samples: int = None,
        max_search_iters: int = None,
    ):
        self.sampling_type = sampling_type
        if self.sampling_type == SamplingType.GRID:
            pass
        elif self.sampling_type == SamplingType.INCREMENTAL:
            assert sample_method is not None
            assert num_samples is not None
            self.sample_method = sample_method
            self.num_samples = num_samples
        elif self.sampling_type == SamplingType.MIN_INCREMENTAL:
            assert sample_method is not None
            assert desired_num_samples is not None
            assert max_search_iters is not None
            self.sample_method = sample_method
            self.desired_num_samples = desired_num_samples
            self.max_search_iters = max_search_iters
            self.step_size = 1


def path_to_joint_solutions(
    path: List[PathPt], robot: Robot, settings: PlanningSetting, scene: Scene
):
    return [pt.to_joint_solutions(robot, settings, scene) for pt in path]

