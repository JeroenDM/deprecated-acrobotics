"""
Wrap the C++ box object as a general python shape.
Other shapes could be added to the C++ code and used by the shape class.

Plotting shapes is implemented here.
In addition the class Colleciont groups shapes with a transforms to define
a shape existing of multiple boxes.

info on collision checking
https://gamedev.stackexchange.com/questions/44500/how-many-and-which-axes-to-use-for-3d-obb-collision-with-sat

@author: jeroen
"""
import numpy as np
from acrobotics.cpp.geometry import Box

class Shape(Box):
    """ Wrapper class for C++ box to enable plotting and transforming the box

    There are two versions of the plot functions, I forgot why.
    It probably has to do with animating stuff.
    """

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
        """ Update existing lines on a plot using the given transform tf as Box position"""
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
        #self.t = tf_shapes
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