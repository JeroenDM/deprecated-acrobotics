import acrobotics.util as util

def test_rot_x():
    R = util.rot_x(0.5)
    assert(R.shape == (3, 3))

def test_pose_x():
    T = util.pose_x(0.5, 1, 2, 3)
    assert(T.shape == (4, 4))