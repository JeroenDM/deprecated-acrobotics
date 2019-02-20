import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.testing import assert_almost_equal
from acrobotics.geometry import Shape
from acrobotics.util import pose_z

tf_identity = np.eye(4)

class TestShape():
    def test_init(self):
        b = Shape(1, 2, 3)

    def test_get_vertices(self):
        b = Shape(1, 2, 3)
        v = b.get_vertices(tf_identity)
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
        n = b.get_normals(tf_identity)
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
        v = b.get_vertices(tf)
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
        v = b.get_vertices(tf)
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
        e = b.get_edges(tf_identity)
        row, col = e.shape
        assert row == 12
        assert col == 6
        v = b.get_vertices(tf_identity)
        # check only one edge
        v0 = np.hstack((v[0], v[1]))
        assert_almost_equal(v0, e[0])

    def test_is_in_collision(self):
        b1 = Shape(1, 1, 1)
        b2 = Shape(1, 1, 2)
        actual = b1.is_in_collision(tf_identity, b2, tf_identity)
        assert actual == True

        b3 = Shape(1, 2, 1)
        T3 = pose_z(np.pi/4, 0.7, 0.7, 0)
        assert b1.is_in_collision(tf_identity, b3, T3) == True

        b4 = Shape(1, 1, 1)
        b5 = Shape(1, 1, 2)
        T4 = pose_z(0, -1, -1, 0)
        T5 = pose_z(np.pi/4, -2, -2, 0)
        assert b4.is_in_collision(T4, b5, T5) == False

    def test_plot(self):
        b1 = Shape(1, 2, 3)
        fig = plt.figure()
        ax = Axes3D(fig)
        b1.plot(ax, tf_identity)
        assert True
