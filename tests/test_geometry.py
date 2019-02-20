import numpy as np
from numpy.testing import assert_almost_equal
from acrobotics.geometry import Shape
from acrobotics.util import pose_z

class TestShape():
    def test_init(self):
        b = Shape(1, 2, 3)

    def test_get_vertices(self):
        b = Shape(1, 2, 3)
        v = b.get_vertices()
        desired = np.array([[-0.5, 1, 1.5],
                           [-0.5, 1, -1.5],
                           [-0.5, -1, 1.5],
                           [-0.5, -1, -1.5],
                           [0.5, 1, 1.5],
                           [0.5, 1, -1.5],
                           [0.5, -1, 1.5],
                           [0.5, -1, -1.5]])
        assert_almost_equal(v, desired)

    def test_get_normals(self):
        b = Shape(1, 2, 3)
        n = b.get_normals()
        desired = np.array([[1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 0, -1]])
        assert_almost_equal(n, desired)

    def test_set_transform(self):
        b = Shape(1, 2, 3)
        tf = np.eye(4)
        tf[0, 3] = 10.5
        b.set_transform(tf)
        v = b.get_vertices()
        desired = np.array([[10, 1, 1.5],
                           [10, 1, -1.5],
                           [10, -1, 1.5],
                           [10, -1, -1.5],
                           [11, 1, 1.5],
                           [11, 1, -1.5],
                           [11, -1, 1.5],
                           [11, -1, -1.5]])
        assert_almost_equal(v, desired)

    def test_set_transform2(self):
        b = Shape(1, 2, 3)
        tf = np.eye(4)
        # rotate pi / 2 around x-axis
        tf[1:3, 1:3] = np.array([[0, -1], [1, 0]])
        b.set_transform(tf)
        v = b.get_vertices()
        desired = np.array([[-0.5, -1.5, 1],
                           [-0.5, 1.5, 1],
                           [-0.5, -1.5, -1],
                           [-0.5, 1.5, -1],
                           [0.5, -1.5, 1],
                           [0.5, 1.5, 1],
                           [0.5, -1.5, -1],
                           [0.5, 1.5, -1]])
        assert_almost_equal(v, desired)

    def test_get_edges(self):
        b = Shape(1, 2, 3)
        e = b.get_edges()
        row, col = e.shape
        assert row == 12
        assert col == 6
        v = b.get_vertices()
        # check only one edge
        v0 = np.hstack((v[0], v[1]))
        assert_almost_equal(v0, e[0])

    def test_is_in_collision(self):
        b1 = Shape(1, 1, 1)
        b2 = Shape(1, 1, 2)
        actual = b1.is_in_collision(b2)
        assert actual == True

        b3 = Shape(1, 2, 1)
        T3 = pose_z(np.pi/4, 0.7, 0.7, 0)
        b3.set_transform(T3)
        assert b1.is_in_collision(b3) == True

        b4 = Shape(1, 1, 1)
        b5 = Shape(1, 1, 2)
        b4.set_transform(pose_z(0, -1, -1, 0))
        b5.set_transform(pose_z(np.pi/4, -2, -2, 0))
        assert b4.is_in_collision(b5) == False
