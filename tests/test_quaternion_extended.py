from acrobotics.pyquat_extended import QuaternionExtended


def test_create_extended_quat():
    q = QuaternionExtended()
    samples = q.random_near(10.0)
    assert isinstance(samples, QuaternionExtended)
