from .types import SamplingType
from .types import SampleMethod


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
