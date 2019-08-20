"""
Defines a general Robot class and implentations of specific robots.
Forward kinematics are included in the general robot.
Inverse kinematics are implemented for specific robots
(3R planar arm and 6DOF Kuka).
"""

import numpy as np
import casadi as ca
from collections import namedtuple
from matplotlib import animation
from .geometry import Collection
from .util import plot_reference_frame

# use named tuples to make data more readable
JointLimit = namedtuple("JointLimit", ["lower", "upper"])
DHLink = namedtuple("DHLink", ["a", "alpha", "d", "theta"])


class Link:
    """ Robot link according to the Denavit-Hartenberg convention. """

    def __init__(self, dh_parameters, joint_type, geometry):
        """ Creates a linkf from Denavit-Hartenberg parameters,
        a joint type ('r' for revolute, 'p' for prismatic) and
        a Collection of Shapes representing the geometry.
        """
        self.dh = dh_parameters
        self.joint_type = joint_type
        self.geometry = geometry

        # save transform object for performance
        self._T = np.eye(4)

    def get_link_relative_transform(self, qi):
        """ transformation matrix from link i relative to i-1

        Links and joints are numbered from 1 to ndof, but python
        indexing of these links goes from 0 to ndof-1!
        """
        if self.joint_type == "r":
            a, alpha, d, theta = self.dh.a, self.dh.alpha, self.dh.d, qi
        if self.joint_type == "p":
            a, alpha, d, theta = self.dh.a, self.dh.alpha, qi, self.dh.theta

        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)
        T = self._T
        T[0, 0], T[0, 1] = c_theta, -s_theta * c_alpha
        T[0, 2], T[0, 3] = s_theta * s_alpha, a * c_theta

        T[1, 0], T[1, 1] = s_theta, c_theta * c_alpha
        T[1, 2], T[1, 3] = -c_theta * s_alpha, a * s_theta

        T[2, 1], T[2, 2], T[2, 3] = s_alpha, c_alpha, d
        return T

    def get_link_relative_transform_casadi(link, qi):
        """ Link transform according to the Denavit-Hartenberg convention.
        Casadi compatible function.
        """
        a, alpha, d, theta = link.dh.a, link.dh.alpha, link.dh.d, qi

        c_t, s_t = ca.cos(theta), ca.sin(theta)
        c_a, s_a = ca.cos(alpha), ca.sin(alpha)

        row1 = ca.hcat([c_t, -s_t * c_a, s_t * s_a, a * c_t])
        row2 = ca.hcat([s_t, c_t * c_a, -c_t * s_a, a * s_t])
        row3 = ca.hcat([0, s_a, c_a, d])
        row4 = ca.hcat([0, 0, 0, 1])

        return ca.vcat([row1, row2, row3, row4])

    def plot(self, ax, tf, *arg, **kwarg):
        self.geometry.plot(ax, tf=tf, *arg, **kwarg)


class Tool(Collection):
    """ Collection with added atribute tool tip transform tf_tt
     relative to the last link.
    """

    def __init__(self, shapes, tf_shapes, tf_tool_tip):
        """
        tf_tool_tip relative to last link robot.
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
        self.joint_limits = [JointLimit(-np.pi, np.pi)] * self.ndof

        # defaul has no fixed base geometry, no tool and
        self.geometry_base = None
        self.tool = None

        # pose of base with respect to the global reference frame
        # this is independent from the geometry of the base,
        # for the whole robot
        self.tf_base = np.eye(4)

        # self collision matrix
        # default: do not check link neighbours, create band structure matrix
        temp = np.ones((self.ndof, self.ndof), dtype="bool")
        self.collision_matrix = np.tril(temp, k=-3) + np.triu(temp, k=3)
        self.do_check_self_collision = True

        # keep track of most likly links to be in collision
        self.collision_priority = list(range(self.ndof))

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
        return T

    def fk_casadi(self, q):
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform_casadi(q[i])
            T = T @ Ti
        if self.tool is not None:
            T = T @ self.tool.tf_tt
        return T

    def fk_all_links(self, q):
        """ Return link frames (not base or tool)
        """
        tf_links = []
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform(q[i])
            T = T @ Ti
            tf_links.append(T)
        return tf_links

    def fk_all_links_casadi(self, q):
        """ Return link frames (not base or tool)
        """
        tf_links = []
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform_casadi(q[i])
            T = T @ Ti
            tf_links.append(T)
        return tf_links

    def _check_self_collision(self, tf_links, geom_links):
        for i, ti, gi in zip(range(self.ndof), tf_links, geom_links):
            for j, tj, gj in zip(range(self.ndof), tf_links, geom_links):
                if self.collision_matrix[i, j]:
                    if gi.is_in_collision(gj, tf_self=ti, tf_other=tj):
                        return True

        # do not check tool against the last link where it is mounted
        if self.tool is not None:
            tf_tool = tf_links[-1]
            for tf_link, geom_link in zip(tf_links[:-1], geom_links[:-1]):
                if geom_link.is_in_collision(
                    self.tool, tf_self=tf_link, tf_other=tf_tool
                ):
                    return True
        return False

    def is_in_self_collision(self, q):
        geom_links = [l.geometry for l in self.links]
        tf_links = self.fk_all_links(q)
        return self._check_self_collision(tf_links, geom_links)

    def is_in_collision(self, q, collection):
        # check collision of fixed base geometry
        base = self.geometry_base
        if base is not None:
            if base.is_in_collision(collection, tf_self=self.tf_base):
                return True

        # check collision for all links
        geom_links = [l.geometry for l in self.links]
        tf_links = self.fk_all_links(q)

        for i in self.collision_priority:
            if geom_links[i].is_in_collision(collection, tf_self=tf_links[i]):
                # move current index to front of priority list
                self.collision_priority.remove(i)
                self.collision_priority.insert(0, i)
                return True

        if self.tool is not None:
            tf_tool = tf_links[-1]
            if self.tool.is_in_collision(collection, tf_self=tf_tool):
                return True

        if self.do_check_self_collision:
            if self._check_self_collision(tf_links, geom_links):
                return True
        return False

    def plot(self, ax, q, *arg, **kwarg):
        if self.geometry_base is not None:
            self.geometry_base.plot(ax, self.tf_base, *arg, **kwarg)

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
        ax.plot(points[:, 0], points[:, 1], points[:, 2], "o-")
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
                for s in l.geometry.shapes:
                    lines.append(s.get_empty_lines(ax, c=(0.1, 0.2, 0.5)))
            if self.tool is not None:
                for s in self.tool.shapes:
                    lines.append(s.get_empty_lines(ax, c=(0.1, 0.2, 0.5)))
            return lines

        def update_lines(frame, q_path, lines):
            tfs = self.fk_all_links(q_path[frame])
            cnt = 0
            for tf_l, l in zip(tfs, self.links):
                for tf_s, s in zip(l.geometry.tf_s, l.geometry.s):
                    Ti = np.dot(tf_l, tf_s)
                    lines[cnt] = s.update_lines(lines[cnt], Ti)
                    cnt = cnt + 1

            if self.tool is not None:
                for tf_s, s in zip(self.tool.tf_s, self.tool.s):
                    tf_j = np.dot(tfs[-1], tf_s)
                    lines[cnt] = s.update_lines(lines[cnt], tf_j)
                    cnt = cnt + 1

        ls = get_emtpy_lines(ax)
        N = len(joint_space_path)
        self.animation = animation.FuncAnimation(
            fig, update_lines, N, fargs=(joint_space_path, ls), interval=200, blit=False
        )
