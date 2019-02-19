"""
Defines a general Robot class and implentations of specific robots.
Forward kinematics are included in the general robot.
Inverse kinematics are implemented for specific robots
(3R planar arm and 6DOF Kuka).
"""

import numpy as np
from collections import namedtuple
from matplotlib import animation
from .geometry import Collection
from .util import plot_reference_frame

# use named tuples to make data more readable
JointLimit = namedtuple('JointLimit', ['lower', 'upper'])
DHLink = namedtuple('DHLink', ['a', 'alpha', 'd', 'theta'])


class Link:
    """ Robot link according to the Denavit-Hartenberg convention. """

    def __init__(self, dh_parameters, joint_type, shape, tf_shape=None):
        self.dh = dh_parameters
        self.shape = shape
        self.joint_type = joint_type
        if tf_shape is None:
            # default value
            self.tf_shape = np.eye(4)
            # move back along x-axis
            # self.tf_rel[0, 3] = -self.dh.a
        else:
            self.tf_shape = tf_shape

    def get_link_relative_transform(self, qi):
        """ transformation matrix from link i relative to i-1

        Links and joints are numbered from 1 to ndof, but python
        indexing of these links goes from 0 to ndof-1!
        """
        if self.joint_type == 'r':
            a, alpha, d, theta = self.dh.a, self.dh.alpha, self.dh.d, qi
        if self.joint_type == 'p':
            a, alpha, d, theta = self.dh.a, self.dh.alpha, qi, self.dh.theta

        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)
        T = np.eye(4)
        T[0, :] = np.array([c_theta,
                            -s_theta * c_alpha,
                            s_theta * s_alpha,
                            a * c_theta])
        T[1, :] = np.array([s_theta,
                            c_theta * c_alpha,
                            -c_theta * s_alpha,
                            a * s_theta])
        T[2, 1:] = np.array([s_alpha, c_alpha, d])
        return T

    def plot(self, ax, tf, *arg, **kwarg):
        tf = np.dot(tf, self.tf_shape)
        self.shape.plot(ax, tf, *arg, **kwarg)


class Tool(Collection):
    """ Group tool shapes and pose
    """

    def __init__(self, shapes, tf_shapes, tf_tool_tip):
        """
        rel_tf is the pose of the tool compared to the robot end-effector
        pose. (Last link)
        tf_tool_tip given relative to rel_tf, saved relative to robot
        end effector frame
        """
        super().__init__(shapes, tf_shapes)
        self.tf_tt = tf_tool_tip


class Robot:
    """ Robot kinematics and shape

    (inital joint values not implemented)
    """

    def __init__(self, links):
        self.links = links
        self.ndof = len(links)

        # set default joint limits
        self.set_joint_limits([JointLimit(-np.pi, np.pi)]*self.ndof)

        # default base position
        self.tf_base = np.eye(4)
        self.shape_base = None

        # no tool by default
        self.tool = None

        # self collision matrix
        # default: do not check neighbours, create band structure matrix
        temp = np.ones((self.ndof, self.ndof), dtype='bool')
        self.collision_matrix = np.tril(temp, k=-3) + np.triu(temp, k=3)
        self.do_check_self_collision = True

        self.tf_tool = None

    def set_joint_limits(self, joint_limits):
        self.joint_limits = joint_limits

    def set_base_tf(self, tf):
        self.tf_base = tf

    def set_base_shape(self, shape, tf):
        self.shape_base = shape
        self.shape_base_tf = tf

    def set_tool(self, tool):
        self.tool = tool

    def fk(self, q):
        """ Return end effector frame, either last link, or tool frame
        if tool available
        """
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform(q[i])
            T = np.dot(T, Ti)
        if self.tool is not None:
            T = np.dot(T, self.tool.tf_tt)
        elif self.tf_tool is not None:
            T = np.dot(T, self.tf_tool)
        return T

    def fk_all_links(self, q):
        """ Return link frames (not base or tool)
        """
        tf_links = []
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform(q[i])
            T = np.dot(T, Ti)
            tf_links.append(T)
        return tf_links

    def get_shapes(self, q):
        tfs = self.fk_all_links(q)
        shapes = [l.shape for l in self.links]
        for i in range(len(shapes)):
            shapes[i].set_transform(tfs[i])

        # add shapes of tool if present
        if self.tool is not None:
            shapes = shapes + self.tool.get_shapes(tf=tfs[-1])
        return shapes

    def check_self_collision(self, s):
        # assume shapes are up to date with the required
        for i in range(self.ndof):
            for j in range(self.ndof):
                if self.collision_matrix[i, j]:
                    if s[i].is_in_collision(s[j]):
                        return True

        # check for collision between tool and robot links
        if self.tool is not None:
            for i in range(self.ndof):
                for j in range(self.ndof, len(s)):
                    if s[i].is_in_collision(s[j]):
                        return True
        return False

    def is_in_collision(self, q, scene):
        # collision between robot and scene
        s_robot = self.get_shapes(q)
        s_scene = scene.get_shapes()
        for sr in s_robot:
            for ss in s_scene:
                if sr.is_in_collision(ss):
                    return True

        # check self-collision if required
        if self.do_check_self_collision:
            if self.check_self_collision(s_robot):
                return True
        return False

    def plot(self, ax, q, *arg, **kwarg):
        # plot base if shape exist
        if self.shape_base is not None:
            tf = np.dot(self.tf_base, self.shape_base_tf)
            self.shape_base.plot(ax, tf, *arg, **kwarg)

        # plot robot linkss
        tf_links = self.fk_all_links(q)
        for i, link in enumerate(self.links):
            link.plot(ax, tf_links[i], *arg, **kwarg)

        if self.tool is not None:
            self.tool.plot(ax, tf=tf_links[-1], *arg, **kwarg)

    def plot_kinematics(self, ax, q, *arg, **kwarg):
        # base frame (0)
        plot_reference_frame(ax, self.tf_base)

        # link frames (1-ndof)
        tf_links = self.fk_all_links(q)
        points = [tf[0:3, 3] for tf in tf_links]
        points = np.array(points)
        points = np.vstack((self.tf_base[0:3, 3], points))
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o-')
        for tfi in tf_links:
            plot_reference_frame(ax, tfi)

        # tool tip frame
        if self.tool is not None:
            tf_tt = np.dot(tf_links[-1], self.tool.tf_tt)
            plot_reference_frame(ax, tf_tt)

    def plot_path(self, ax, joint_space_path):
        alpha = np.linspace(1, 0.2, len(joint_space_path))
        for i, qi in enumerate(joint_space_path):
            self.plot(ax, qi, c=(0.1, 0.2, 0.5, alpha[i]))

    def animate_path(self, fig, ax, joint_space_path):
        def get_emtpy_lines(ax):
            lines = []
            for l in self.links:
                lines.append(l.shape.get_empty_lines(ax, c=(0.1, 0.2, 0.5)))
            if self.tool is not None:
                for s in self.tool.s:
                    lines.append(s.get_empty_lines(ax, c=(0.1, 0.2, 0.5)))
            return lines

        def update_lines(frame, q_path, lines):
            tfs = self.fk_all_links(q_path[frame])
            N_links = len(self.links)
            for i in range(N_links):
                Ti = np.dot(tfs[i], self.links[i].tf_shape)
                lines[i] = self.links[i].shape.update_lines(lines[i], Ti)
            if self.tool is not None:
                N_s = len(self.tool.s)
                for j, k in zip(range(N_links, N_links + N_s), range(N_s)):
                    tf_j = np.dot(tfs[-1], self.tool.tf_s[k])
                    lines[j] = self.tool.s[k].update_lines(lines[j], tf_j)

        ls = get_emtpy_lines(ax)
        N = len(joint_space_path)
        self.animation = animation.FuncAnimation(fig, update_lines, N,
                                                 fargs=(joint_space_path, ls),
                                                 interval=200, blit=False)
