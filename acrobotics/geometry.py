import numpy as np
import fcl


def tf_apply(tf, vector):
    """ Transform a vector with a homogeneous transform matrix tf. """
    return np.dot(tf[:3, :3], vector) + tf[:3, 3]


class Shape:
    """ Just a Box with three sides for now.

    A Shape has no inherent position! You always have to specify
    a transform when you want something from a shape.
    It is straigh forward to implement fixed shapes for objects
    that do not move. But not really a performance issue for now.

    Wraps around an fcl_shape for collision checking.
    Generated vertices and edges for plotting.
    """

    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.fcl_shape = fcl.Box(dx, dy, dz)
        self.request = fcl.CollisionRequest()
        self.result = fcl.CollisionResult()

    def get_vertices(self, tf):
        v = np.zeros((8, 3))
        a = self.dx / 2
        b = self.dy / 2
        c = self.dz / 2

        v[0] = tf_apply(tf, [-a,  b,  c])
        v[1] = tf_apply(tf, [-a,  b, -c])
        v[2] = tf_apply(tf, [-a, -b,  c])
        v[3] = tf_apply(tf, [-a, -b, -c])

        v[4] = tf_apply(tf, [a,  b,  c])
        v[5] = tf_apply(tf, [a,  b, -c])
        v[6] = tf_apply(tf, [a, -b,  c])
        v[7] = tf_apply(tf, [a, -b, -c])
        return v

    def get_edges(self, tf):
        v = self.get_vertices(tf)
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

    def get_normals(self, tf):
        n = np.zeros((6, 3))
        R = tf[:3, :3]
        n[0] = np.dot(R, [ 1,  0,  0])
        n[1] = np.dot(R, [-1,  0,  0])
        n[2] = np.dot(R, [ 0,  1,  0])
        n[3] = np.dot(R, [ 0, -1,  0])
        n[4] = np.dot(R, [ 0,  0,  1])
        n[5] = np.dot(R, [ 0,  0, -1])
        return n

    def is_in_collision(self, tf, other, tf_other):
        fcl_tf_1 = fcl.Transform(tf[:3, :3], tf[:3, 3])
        fcl_tf_2 = fcl.Transform(tf_other[:3, :3], tf_other[:3, 3])

        o1 = fcl.CollisionObject(self.fcl_shape, fcl_tf_1)
        o2 = fcl.CollisionObject(other.fcl_shape, fcl_tf_2)

        return fcl.collide(o1, o2, self.request, self.result)

    def is_in_collision_multi(self, others):
        for other in others:
            if self.is_in_collision(other):
                return True
        return False

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
        edges = self.get_edges(tf)
        for i, l in enumerate(lines):
            x = [edges[i, 0], edges[i, 3]]
            y = [edges[i, 1], edges[i, 4]]
            z = [edges[i, 2], edges[i, 5]]
            l.set_data(x, y)
            l.set_3d_properties(z)
        return lines


class Collection:
    """ Group Shapes and transforms for the shapes.
    This is used to create more complicated objects that just a box.
    """

    def __init__(self, shapes, tf_shapes):
        self.s = shapes
        self.tf_s = tf_shapes

    def plot(self, ax, *arg, **kwarg):
        if 'tf' in kwarg:
            tf = kwarg.pop('tf')
            for shape, tf_shape in zip(self.s, self.tf_s):
                shape.plot(ax, np.dot(tf, tf_shape), *arg, **kwarg)
        else:
            for shape, tf_shape in zip(self.s, self.tf_s):
                shape.plot(ax, tf_shape,  *arg, **kwarg)

    @property
    def shapes(self):
        return self.s

    def is_in_collision(self, other, tf_self=None, tf_other=None):
        tf_shapes_self = self.tf_s
        tf_shapes_other = other.tf_s

        # move the collection of shapes if specified
        if tf_self is not None:
            tf_shapes_self = [np.dot(tf_self, tf) for tf in tf_shapes_self]
        if tf_other is not None:
            tf_shapes_other = [np.dot(tf_other, tf) for tf in tf_shapes_other]

        # check for collision between all those shapes
        for tf1, shape1 in zip(tf_shapes_self, self.s):
            for tf2, shape2 in zip(tf_shapes_other, other.s):
                if shape1.is_in_collision(tf1, shape2, tf2):
                    return True
        return False
