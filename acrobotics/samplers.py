import numpy as np
from numpy.random import uniform
from .pyquat_extended import QuaternionExtended as Quaternion
from enum import Enum


class SampleMethod(Enum):
    random_uniform = 0
    deterministic_uniform = 1

class Sampler:
    def __init__(self):
        self.halton_sampler: HaltonSampler = None

    def sample(self, num_samples: int, sample_dim: int, method: SampleMethod):
        if method == SampleMethod.random_uniform:
            return np.random.rand(num_samples, sample_dim)
        elif method == SampleMethod.deterministic_uniform:
            if self.halton_sampler is None:
                self.halton_sampler = HaltonSampler(sample_dim)
            # make sure an existing sampler has the correct dimension
            assert self.halton_sampler.dim == sample_dim
            return self.halton_sampler.get_samples(num_samples)
        else:
            raise NotImplementedError(f"Unkown sampling method: {method}")


def vdc(n, base=2):
    """ Create van der Corput sequence

    source for van der Corput and Halton sampling code
    https://laszukdawid.com/2017/02/04/halton-sequence-in-python/
    """
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc


def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2, int(num ** 0.5) + 1):
            if (num % i) == 0:
                return False
        return True

    prime = 3
    while 1:
        if is_prime(prime):
            yield prime
        prime += 2


class HaltonSampler:
    def __init__(self, dim):
        self.dim = dim

        # setup primes for every dimension
        prime_factory = next_prime()
        self.primes = []
        for _ in range(dim):
            self.primes.append(next(prime_factory))

        # init counter for van der Corput sampling
        self.cnt = 1

    def get_samples(self, n):
        seq = []
        for d in range(self.dim):
            base = self.primes[d]
            seq.append([vdc(i, base) for i in range(self.cnt, self.cnt + n)])
        self.cnt += n
        return np.array(seq).T


def sample_SO3(n=10, rep="rpy", method="random"):
    """Generate a random unit quaternion.

    Uniformly distributed across the rotation space
    As per: http://planning.cs.uiuc.edu/node198.html
    and code from http://kieranwynn.github.io/pyquaternion
    """
    if method is "random":
        r1, r2, r3 = np.random.rand(3, n)
    elif method is "halton":
        hs = HaltonSampler(3)
        r1, r2, r3 = (hs.get_samples(n)).T
    else:
        raise ValueError("Invalid sampling method: {}".format(method))

    q1 = np.sqrt(1.0 - r1) * (np.sin(2 * np.pi * r2))
    q2 = np.sqrt(1.0 - r1) * (np.cos(2 * np.pi * r2))
    q3 = np.sqrt(r1) * (np.sin(2 * np.pi * r3))
    q4 = np.sqrt(r1) * (np.cos(2 * np.pi * r3))

    result = [Quaternion(q1[i], q2[i], q3[i], q4[i]) for i in range(n)]
    if rep == "quat":
        return result
    elif rep == "transform":
        return [q.transformation_matrix for q in result]
    elif rep == "rpy":
        return [np.array(q.yaw_pitch_roll) for q in result]
