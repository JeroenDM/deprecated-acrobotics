import numpy as np
import fcl


def tf_apply(tf, vector):
    """ Transform a vector with a homogeneous transform matrix tf. """
    return np.dot(tf[:3, :3], vector) + tf[:3, 3]


class Shape:
    """ Wrapper class for fcl collision objects
    """

    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.tf = np.eye(4)
        self.fcl_shape = fcl.Box(dx, dy, dz)

    def set_transform(self, new_tf):
        self.tf = new_tf

    def get_vertices(self):
        v = np.zeros((8, 3))
        a = self.dx / 2
        b = self.dy / 2
        c = self.dz / 2

        v[0] = tf_apply(self.tf, [-a,  b,  c])
        v[1] = tf_apply(self.tf, [-a,  b, -c])
        v[2] = tf_apply(self.tf, [-a, -b,  c])
        v[3] = tf_apply(self.tf, [-a, -b, -c])

        v[4] = tf_apply(self.tf, [a,  b,  c])
        v[5] = tf_apply(self.tf, [a,  b, -c])
        v[6] = tf_apply(self.tf, [a, -b,  c])
        v[7] = tf_apply(self.tf, [a, -b, -c])
        return v

    def get_edges(self):
        v = self.get_vertices()
        e = np.zeros((12, 6))
        e[0] = np.hstack((v[0], v[1]))
        e[1] = np.hstack((v[1], v[3]))
        e[2] = np.hstack((v[3], v[2]))
        e[3] = np.hstack((v[2], v[0]))

        e[4] = np.hstack((v[0], v[4]))
        e[5] = np.hstack((v[1], v[5]))
        e[6] = np.hstack((v[3], v[7]))
        e[7] = np.hstack((v[2], v[6]))

        e[8] = np.hstack((v[4], v[5]))
        e[9] = np.hstack((v[5], v[7]))
        e[10] = np.hstack((v[7], v[6]))
        e[11] = np.hstack((v[6], v[4]))
        return e

    def get_normals(self):
        n = np.zeros((6, 3))
        R = self.tf[:3, :3]
        n[0] = np.dot(R, [ 1,  0,  0])
        n[1] = np.dot(R, [-1,  0,  0])
        n[2] = np.dot(R, [ 0,  1,  0])
        n[3] = np.dot(R, [ 0, -1,  0])
        n[4] = np.dot(R, [ 0,  0,  1])
        n[5] = np.dot(R, [ 0,  0, -1])
        return n

    def is_in_collision(self, other):
        fcl_tf_1 = fcl.Transform(self.tf[:3, :3], self.tf[:3, 3])
        fcl_tf_2 = fcl.Transform(other.tf[:3, :3], other.tf[:3, 3])
        o1 = fcl.CollisionObject(self.fcl_shape, fcl_tf_1)
        o2 = fcl.CollisionObject(other.fcl_shape, fcl_tf_2)
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        result = fcl.collide(o1, o2, request, result)
        return result

    def plot(self, ax, tf, *arg, **kwarg):
        """ Plot a box as lines on a given axes_handle.

        >>> from acrobotics.util import get_default_axes3d
        >>> fig, ax = get_default_axes3d()
        >>> shape = Shape(0.5, 0.8, 0.9)
        >>> tf = np.eye(4)
        >>> shape.plot(ax, tf)

        """
        lines = self.get_empty_lines(ax, *arg, **kwarg)
        lines = self.update_lines(lines, tf)

    def get_empty_lines(self, ax, *arg, **kwarg):
        """ Create empty lines to initialize an animation """
        return [ax.plot([], [], '-', *arg, **kwarg)[0] for i in range(12)]

    def update_lines(self, lines, tf):
        """ Update existing lines on a plot using the given transform tf"""
        self.set_transform(tf)
        edges = self.get_edges()
        for i, l in enumerate(lines):
            x = [edges[i, 0], edges[i, 3]]
            y = [edges[i, 1], edges[i, 4]]
            z = [edges[i, 2], edges[i, 5]]
            l.set_data(x, y)
            l.set_3d_properties(z)
        return lines

    def plot_2(self, ax, *arg, **kwarg):
        lines = self.get_empty_lines(ax, *arg, **kwarg)
        lines = self.update_lines_2(lines)

    def update_lines_2(self, lines):
        edges = self.get_edges()
        for i, l in enumerate(lines):
            x = [edges[i, 0], edges[i, 3]]
            y = [edges[i, 1], edges[i, 4]]
            z = [edges[i, 2], edges[i, 5]]
            l.set_data(x, y)
            l.set_3d_properties(z)
        return lines


class Collection:
    """ shapes and there transforms in one class
    """

    def __init__(self, shapes, tf_shapes):
        self.s = shapes
        self.tf_s = tf_shapes
        for shape, tf in zip(self.s, tf_shapes):
            shape.set_transform(tf)

    def plot(self, ax, *arg, **kwarg):
        if 'tf' in kwarg:
            tf = kwarg.pop('tf')
            for i, shape in enumerate(self.s):
                shape.set_transform(np.dot(tf, self.tf_s[i]))
                shape.plot_2(ax, *arg, **kwarg)
        else:
            for shape in self.s:
                shape.plot_2(ax, *arg, **kwarg)

    def get_shapes(self, tf=None):
        if tf is None:
            return self.s
        else:
            for i, shape in enumerate(self.s):
                shape.set_transform(np.dot(tf, self.tf_s[i]))
            return self.s
