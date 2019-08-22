import numpy as np
from ..robot import Robot, DHLink, Link, IKResult
from acrobotics.geometry import Shape, Scene
from ..util import pose_x, tf_inverse, translation

pi = np.pi


class PlanarArm(Robot):
    """ Robot defined on page 69 in book Siciliano """

    def __init__(self, a1=1, a2=1, a3=1):
        geometry = [
            Scene([Shape(-a1 / 2, 0.1, 0.1)], [np.eye(4)]),
            Scene([Shape(-a2 / 2, 0.1, 0.1)], [np.eye(4)]),
            Scene([Shape(-a3 / 2, 0.1, 0.1)], [np.eye(4)]),
        ]
        super().__init__(
            [
                Link(DHLink(a1, 0, 0, 0), "r", geometry[0]),
                Link(DHLink(a2, 0, 0, 0), "r", geometry[1]),
                Link(DHLink(a3, 0, 0, 0), "r", geometry[2]),
            ]
        )


class SphericalArm(Robot):
    """ Robot defined on page 72 in book Siciliano """

    def __init__(self, d2=1):
        geometry = [
            Scene([Shape(1, 0.1, 0.1)], [np.eye(4)]),
            Scene([Shape(1, 0.1, 0.1)], [np.eye(4)]),
            Scene([Shape(1, 0.1, 0.1)], [np.eye(4)]),
        ]
        super().__init__(
            [
                Link(DHLink(0, -pi / 2, 0, 0), "r", geometry[0]),
                Link(DHLink(0, pi / 2, d2, 0), "r", geometry[1]),
                Link(DHLink(0, 0, 0, 0), "p", geometry[2]),
            ]
        )


class SphericalWrist(Robot):
    """ Robot defined on page 75 in book Siciliano """

    def __init__(self, d3=1):
        geometry = [
            Scene([Shape(1, 0.1, 0.1)], [np.eye(4)]),
            Scene([Shape(1, 0.1, 0.1)], [np.eye(4)]),
            Scene([Shape(1, 0.1, 0.1)], [np.eye(4)]),
        ]
        super().__init__(
            [
                Link(DHLink(0, -pi / 2, 0, 0), "r", geometry[0]),
                Link(DHLink(0, pi / 2, 0, 0), "r", geometry[1]),
                Link(DHLink(0, 0, d3, 0), "r", geometry[2]),
            ]
        )

        # cache ik solution array object for performance
        self.qsol = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def ik(self, T) -> IKResult:
        """ TODO add base frame correction
        """
        Rbase = self.tf_base[:3, :3]
        Ree = T[:3, :3]
        Ree_rel = np.dot(Rbase.transpose(), Ree)
        # ignore position
        # n s a according to convention Siciliano
        n = Ree_rel[:3, 0]
        s = Ree_rel[:3, 1]
        a = Ree_rel[:3, 2]

        A = np.sqrt(a[0] ** 2 + a[1] ** 2)
        # solution with theta2 in (0, pi)
        t1_1 = np.arctan2(a[1], a[0])
        t2_1 = np.arctan2(A, a[2])
        t3_1 = np.arctan2(s[2], -n[2])
        # solution with theta2 in (-pi, 0)
        t1_2 = np.arctan2(-a[1], -a[0])
        t2_2 = np.arctan2(-A, a[2])
        t3_2 = np.arctan2(-s[2], n[2])

        q_sol = self.qsol
        q_sol[0, 0], q_sol[0, 1], q_sol[0, 2] = t1_1, t2_1, t3_1
        q_sol[1, 0], q_sol[1, 1], q_sol[1, 2] = t1_2, t2_2, t3_2
        return IKResult(True, q_sol)


class AnthropomorphicArm(Robot):
    """ Robot defined on page 73 in book Siciliano """

    def __init__(self, a2=1, a3=1):
        geometry = [
            Scene([Shape(0.1, 0.3, 0.3)], [translation(-0.05, 0, 0)]),
            Scene([Shape(a2, 0.1, 0.1)], [translation(-a2 / 2, 0, 0)]),
            Scene([Shape(a3, 0.1, 0.1)], [translation(-a3 / 2, 0, 0)]),
        ]
        super().__init__(
            [
                Link(DHLink(0, pi / 2, 0, 0), "r", geometry[0]),
                Link(DHLink(a2, 0, 0, 0), "r", geometry[1]),
                Link(DHLink(a3, 0, 0, 0), "r", geometry[2]),
            ]
        )

    def ik(self, T) -> IKResult:
        # ignore orientation
        px, py, pz = T[0, 3], T[1, 3], T[2, 3]
        l2, l3 = self.links[1].dh.a, self.links[2].dh.a
        tol = 1e-6
        Lps = px ** 2 + py ** 2 + pz ** 2

        # Theta 3
        # =======
        c3 = (Lps - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)

        if c3 > (1 + tol) or c3 < -(1 + tol):
            return IKResult(False)
        elif c3 > (1 - tol) or c3 < -(1 - tol):
            # almost 1 or -1 => fix it
            c3 = np.sign(c3)

        # point should be reachable
        # TODO just use arccos
        s3 = np.sqrt(1 - c3 ** 2)

        t3_1, t3_2 = np.arctan2(s3, c3), np.arctan2(-s3, c3)

        # Theta 2
        # =======
        # TODO must be greater than zero (numerical error?)
        Lxy = np.sqrt(px ** 2 + py ** 2)
        A = l2 + l3 * c3  # often used expression
        B = l3 * s3
        # positive sign for s3
        t2_1 = np.arctan2(pz * A - Lxy * B, Lxy * A + pz * B)
        t2_2 = np.arctan2(pz * A + Lxy * B, -Lxy * A + pz * B)
        # negative sign for s3
        t2_3 = np.arctan2(pz * A + Lxy * B, Lxy * A - pz * B)
        t2_4 = np.arctan2(pz * A - Lxy * B, -Lxy * A - pz * B)

        # Theta 1
        # =======
        t1_1 = np.arctan2(py, px)
        t1_2 = np.arctan2(-py, -px)

        # 4 solutions
        # =========
        q_sol = [
            [t1_1, t2_1, t3_1],
            [t1_1, t2_3, t3_2],
            [t1_2, t2_2, t3_1],
            [t1_2, t2_4, t3_2],
        ]

        return IKResult(True, q_sol)


class Arm2(Robot):
    """ Articulated arm with first link length is NOT zeros
    In addition the last frame is rotated to get the Z axis pointing out
    along the hypothetical 4 joint when adding a wrist"""

    def __init__(self, a1=1, a2=1, a3=1):
        geometry = [
            Scene([Shape(a1, 0.1, 0.1)], [np.eye(4)]),
            Scene([Shape(a2, 0.1, 0.1)], [np.eye(4)]),
            Scene([Shape(a3, 0.1, 0.1)], [np.eye(4)]),
        ]
        super().__init__(
            [
                Link(DHLink(a1, pi / 2, 0, 0), "r", geometry[0]),
                Link(DHLink(a2, 0, 0, 0), "r", geometry[1]),
                Link(DHLink(a3, pi / 2, 0, 0), "r", geometry[2]),
            ]
        )

    def ik(self, T) -> IKResult:
        # compensate for tool frame
        #    if self.tf_tool is not None:
        #        print('Arm2: adapt for tool')
        #        print(T)
        #        Tw = np.dot(T, tf_inverse(self.tf_tool))
        #        p = Tw[:3, 3]
        #        print(Tw)
        #    else:
        p = T[:3, 3]
        # ignore orientation
        x, y, z = p[0], p[1], p[2]
        a1, a2, a3 = self.links[0].dh.a, self.links[1].dh.a, self.links[2].dh.a
        tol = 1e-6
        reachable_pos = True
        reachable_neg = True

        # Theta 1
        # =======
        q1_pos = np.arctan2(y, x)
        q1_neg = np.arctan2(-y, -x)

        # Theta 3
        # =======
        # q3 two options, elbow up and elbow down
        # return solutions between in interval (-pi, pi)
        den = 2 * a2 * a3
        num = a1 ** 2 - a2 ** 2 - a3 ** 2 + x ** 2 + y ** 2 + z ** 2

        c1, s1 = np.cos(q1_pos), np.sin(q1_pos)
        c3 = (num - 2 * a1 * s1 * y - 2 * a1 * x * c1) / den

        if c3 > (1 + tol) or c3 < -(1 + tol):
            reachable_pos = False
        else:
            if c3 > (1 - tol) or c3 < -(1 - tol):
                # almost 1 or -1 => fix it
                c3 = np.sign(c3)

            s3 = np.sqrt(1 - c3 ** 2)
            q3_pos_a = np.arctan2(s3, c3)
            q3_pos_b = np.arctan2(-s3, c3)

        c1, s1 = np.cos(q1_neg), np.sin(q1_neg)
        c3 = (num - 2 * a1 * s1 * y - 2 * a1 * x * c1) / den

        if c3 > (1 + tol) or c3 < -(1 + tol):
            reachable_neg = False
        else:
            if c3 > (1 - tol) or c3 < -(1 - tol):
                # almost 1 or -1 => fix it
                c3 = np.sign(c3)

            s3 = np.sqrt(1 - c3 ** 2)
            q3_neg_a, q3_neg_b = np.arctan2(s3, c3), np.arctan2(-s3, c3)

        # Theta 2
        # =======
        L = np.sqrt(x ** 2 + y ** 2)
        if reachable_pos:
            c3, s3 = np.cos(q3_pos_b), np.sin(q3_pos_a)

            q2_pos_a = np.arctan2(
                (-a3 * s3 * (L - a1) + z * (a2 + a3 * c3)),
                (a3 * s3 * z + (L - a1) * (a2 + a3 * c3)),
            )
            q2_pos_b = np.arctan2(
                (a3 * s3 * (L - a1) + z * (a2 + a3 * c3)),
                (-a3 * s3 * z + (L - a1) * (a2 + a3 * c3)),
            )
        if reachable_neg:
            s3 = np.sin(q3_neg_a)
            c3 = np.cos(q3_neg_b)
            q2_neg_a = np.arctan2(
                (a3 * s3 * (L + a1) + z * (a2 + a3 * c3)),
                (a3 * s3 * z - (L + a1) * (a2 + a3 * c3)),
            )
            q2_neg_b = np.arctan2(
                (-a3 * s3 * (L + a1) + z * (a2 + a3 * c3)),
                (-a3 * s3 * z - (L + a1) * (a2 + a3 * c3)),
            )
        # q2_neg_a = -q2_neg_a
        # q2_neg_b = -q2_neg_b

        # 4 solutions
        # =========
        #        q_sol = [[q1_pos, q2_pos_a, q3_a],
        #                 [q1_pos, q2_pos_b, q3_b]]
        q_sol = []
        if reachable_pos:
            q_sol.append([q1_pos, q2_pos_a, q3_pos_a])
            q_sol.append([q1_pos, q2_pos_b, q3_pos_b])
        if reachable_neg:
            q_sol.append([q1_neg, q2_neg_a, q3_neg_a])
            q_sol.append([q1_neg, q2_neg_b, q3_neg_b])

        if reachable_pos or reachable_neg:
            return IKResult(True, q_sol)
        else:
            return IKResult(False)


class Kuka(Robot):
    """ Robot combining AnthropomorphicArm and SphericalWrist
    """

    def __init__(self, a1=0.18, a2=0.6, d4=0.62, d6=0.115):
        # define kuka collision shapes
        s = [
            Shape(0.3, 0.2, 0.1),
            Shape(0.8, 0.2, 0.1),
            Shape(0.2, 0.1, 0.5),
            Shape(0.1, 0.2, 0.1),
            Shape(0.1, 0.1, 0.085),
            Shape(0.1, 0.1, 0.03),
        ]
        # define transforms for collision shapes
        tfs = [
            pose_x(0, -0.09, 0, 0.05),
            pose_x(0, -0.3, 0, -0.05),
            pose_x(0, 0, 0.05, 0.17),
            pose_x(0, 0, 0.1, 0),
            pose_x(0, 0, 0, 0.085 / 2),
            pose_x(0, 0, 0, -0.03 / 2),
        ]

        # create robot
        super().__init__(
            [
                Link(DHLink(a1, pi / 2, 0, 0), "r", Scene([s[0]], [tfs[0]])),
                Link(DHLink(a2, 0, 0, 0), "r", Scene([s[1]], [tfs[1]])),
                Link(DHLink(0, pi / 2, 0, 0), "r", Scene([s[2]], [tfs[2]])),
                Link(DHLink(0, -pi / 2, d4, 0), "r", Scene([s[3]], [tfs[3]])),
                Link(DHLink(0, pi / 2, 0, 0), "r", Scene([s[4]], [tfs[4]])),
                Link(DHLink(0, 0, d6, 0), "r", Scene([s[5]], [tfs[5]])),
            ]
        )
        self.arm = Arm2(a1=a1, a2=a2, a3=d4)
        self.wrist = SphericalWrist(d3=d6)

    def ik(self, T) -> IKResult:
        # copy transform to change it without affecting the T given
        Tw = T.copy()
        # compensate for base
        Tw = np.dot(tf_inverse(self.tf_base), Tw)
        # compensate for tool frame
        if self.tool is not None:
            Tw = np.dot(Tw, tf_inverse(self.tool.tf_tt))
        # compensate for d6, last link length
        d6 = self.links[5].dh.d
        v6 = np.dot(Tw[:3, :3], np.array([0, 0, d6]))
        Tw[:3, 3] = Tw[:3, 3] - v6
        sol_arm = self.arm.ik(Tw)
        if sol_arm.success:
            solutions = []
            for q_arm in sol_arm.solutions:
                q_arm[2] = q_arm[2] + pi / 2
                base_wrist = np.eye(4)
                # get orientation from arm fk
                base_wrist[:3, :3] = self.arm.fk(q_arm)[:3, :3]
                # position from given T (not used actually)
                base_wrist[3, :3] = Tw[3, :3]
                self.wrist.tf_base = base_wrist
                sol_wrist = self.wrist.ik(Tw)
                if sol_wrist.success:
                    for q_wrist in sol_wrist.solutions:
                        solutions.append(np.hstack((q_arm, q_wrist)))
            if len(solutions) > 0:
                return IKResult(True, solutions)
        return IKResult(False)


class KukaOnRail(Robot):
    def __init__(self, a1=0.18, a2=0.6, d4=0.62, d6=0.115):
        s = [
            Shape(0.2, 0.2, 0.1),
            Shape(0.3, 0.2, 0.1),
            Shape(0.8, 0.2, 0.1),
            Shape(0.2, 0.1, 0.5),
            Shape(0.1, 0.2, 0.1),
            Shape(0.1, 0.1, 0.085),
            Shape(0.1, 0.1, 0.03),
        ]

        tfs = [
            pose_x(0, 0, 0, -0.15),
            pose_x(0, -0.09, 0, 0.05),
            pose_x(0, -0.3, 0, -0.05),
            pose_x(0, 0, 0.05, 0.17),
            pose_x(0, 0, 0.1, 0),
            pose_x(0, 0, 0, 0.085 / 2),
            pose_x(0, 0, 0, -0.03 / 2),
        ]
        # create robot
        super().__init__(
            [
                Link(DHLink(0, pi / 2, 0, 0), "p", Scene([s[0]], [tfs[0]])),
                Link(DHLink(a1, pi / 2, 0, 0), "r", Scene([s[1]], [tfs[1]])),
                Link(DHLink(a2, 0, 0, 0), "r", Scene([s[2]], [tfs[2]])),
                Link(DHLink(0, pi / 2, 0, 0), "r", Scene([s[3]], [tfs[3]])),
                Link(DHLink(0, -pi / 2, d4, 0), "r", Scene([s[4]], [tfs[4]])),
                Link(DHLink(0, pi / 2, 0, 0), "r", Scene([s[5]], [tfs[5]])),
                Link(DHLink(0, 0, d6, 0), "r", Scene([s[6]], [tfs[6]])),
            ]
        )
        self.kuka = Kuka(a1=0.18, a2=0.6, d4=0.62, d6=0.115)

    def ik(self, T, q_fixed) -> IKResult:
        # copy transform to change it without affecting the T given
        Tw = T.copy()
        # compensate for base
        Tw = np.dot(tf_inverse(self.tf_base), Tw)
        # compensate for tool frame
        if self.tool is not None:
            Tw = np.dot(Tw, tf_inverse(self.tool.tf_tt))

        # change base of helper robot according to fixed joint
        T1 = self.links[0].get_link_relative_transform(q_fixed)
        self.kuka.tf_base = T1

        res = self.kuka.ik(Tw)
        if res.success:
            q_sol = []
            for qi in res.solutions:
                q_sol.append(np.hstack((q_fixed, qi)))
            return IKResult(True, q_sol)
        else:
            return IKResult(False)


class Puma(Robot):
    """ Puma parameters from https://github.com/uw-biorobotics/IKBT

    Note that I use the convention of Siciliano for the a index for
    a3 and a4

    ik code was in the old version, but the code was from someone else
    and I did not use it. TODO add link here.
    TODO fix geometry for plotting and collision checking.
    """

    def __init__(self, a3=0.432, a4=0.0203, d1=0.6, d3=0.1245, d4=0.432):
        geometry = [
            Scene(Shape(0.1, 0.1, 0.1), np.eye(4)),
            Scene(Shape(0.1, 0.1, 0.1), np.eye(4)),
            Scene(Shape(0.1, 0.1, 0.1), np.eye(4)),
            Scene(Shape(0.1, 0.1, 0.1), np.eye(4)),
            Scene(Shape(0.1, 0.1, 0.1), np.eye(4)),
            Scene(Shape(0.1, 0.1, 0.1), np.eye(4)),
        ]
        super().__init__(
            [
                Link(DHLink(0, 0, d1, 0), "r", geometry[0]),
                Link(DHLink(0, -pi / 2, 0, 0), "r", geometry[1]),
                Link(DHLink(a3, 0, d3, 0), "r", geometry[2]),
                Link(DHLink(a4, -pi / 2, d4, 0), "r", geometry[3]),
                Link(DHLink(0, -pi / 2, 0, 0), "r", geometry[4]),
                Link(DHLink(0, pi / 2, 0, 0), "r", geometry[5]),
            ]
        )
