"""
Defines a general Robot class and implentations of specific robots.
Forward kinematics are included in the general robot.
Inverse kinematics are implemented for specific robots
(3R planar arm and 6DOF Kuka).
"""

import numpy as np
from numpy import cos, sin, array, pi, sqrt, arctan2, sign
from collections import namedtuple
from matplotlib import animation
from .geometry import Shape, Collection
from .util import tf_inverse, pose_x, plot_reference_frame

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
            #self.tf_rel[0, 3] = -self.dh.a
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
        
        c_theta = cos(theta)
        s_theta = sin(theta)
        c_alpha = cos(alpha)
        s_alpha = sin(alpha)
        T = np.eye(4)
        T[0, :] = array([c_theta,
                         -s_theta * c_alpha,
                         s_theta * s_alpha,
                         a * c_theta]) 
        T[1, :] = array([s_theta,
                         c_theta * c_alpha,
                         -c_theta * s_alpha,
                         a * s_theta])
        T[2, 1:] = array([s_alpha, c_alpha, d])
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
        # default: do not check neighbours
        temp = np.ones((self.ndof, self.ndof), dtype='bool')
        self.collision_matrix = np.tril(temp, k=-3) + np.triu(temp, k=3)
        #self.collision_matrix = np.zeros((self.ndof, self.ndof), dtype='bool')
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
        s_scene  = scene.get_shapes()
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
        
        l = get_emtpy_lines(ax)
        N = len(joint_space_path)
        self.animation = animation.FuncAnimation(fig, update_lines, N,
                                           fargs=(joint_space_path, l),
                                           interval=200, blit=False)

class PlanarArm(Robot):
    """ Robot defined on page 69 in book Siciliano """
    def __init__(self, a1=1, a2=1, a3=1):
        super().__init__(
                [Link(DHLink(a1, 0, 0, 0), 'r', Shape(a1, 0.1, 0.1)),
                 Link(DHLink(a2, 0, 0, 0), 'r', Shape(a2, 0.1, 0.1)),
                 Link(DHLink(a3, 0, 0, 0), 'r', Shape(a3, 0.1, 0.1))]
                )

class SphericalArm(Robot):
    """ Robot defined on page 72 in book Siciliano """
    def __init__(self, d2=1):
        super().__init__(
                [Link(DHLink(0, -pi/2, 0 , 0), 'r', Shape(1, 0.1, 0.1)),
                 Link(DHLink(0,  pi/2, d2, 0), 'r', Shape(1, 0.1, 0.1)),
                 Link(DHLink(0,  0   , 0 , 0), 'p', Shape(1, 0.1, 0.1))]
                )

class SphericalWrist(Robot):
    """ Robot defined on page 75 in book Siciliano """
    def __init__(self, d3=1):
        super().__init__(
                [Link(DHLink(0, -pi/2, 0,  0), 'r', Shape(1, 0.1, 0.1)),
                 Link(DHLink(0,  pi/2, 0,  0), 'r', Shape(1, 0.1, 0.1)),
                 Link(DHLink(0,  0   , d3, 0), 'r', Shape(1, 0.1, 0.1))]
                )
    
    def ik(self, T):
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
        
        A = sqrt(a[0]**2 + a[1]**2)
        # solution with theta2 in (0, pi)
        t1_1 = arctan2(a[1], a[0])
        t2_1 = arctan2(A, a[2])
        t3_1 = arctan2(s[2], -n[2])
        # solution with theta2 in (-pi, 0)
        t1_2 = arctan2(-a[1], -a[0])
        t2_2 = arctan2(-A, a[2])
        t3_2 = arctan2(-s[2], n[2])
        
        qsol = np.array([[t1_1, t2_1, t3_1],
                         [t1_2, t2_2, t3_2]])
        return {'success': True,
                'sol': qsol}

class AnthropomorphicArm(Robot):
    """ Robot defined on page 73 in book Siciliano """
    def __init__(self, a2=1, a3=1):
        super().__init__(
                [Link(DHLink(0 , pi/2, 0, 0), 'r', Shape(0.1, 0.1, 0.1)),
                 Link(DHLink(a2, 0   , 0, 0), 'r', Shape(a2, 0.1, 0.1)),
                 Link(DHLink(a3, 0   , 0, 0), 'r', Shape(a3, 0.1, 0.1))]
                )
    
    def ik(self, T):
        # ignore orientation
        px, py, pz = T[0, 3], T[1, 3], T[2, 3]
        l1, l2, l3 = self.links[0].dh.a, self.links[1].dh.a, self.links[2].dh.a
        tol = 1e-6
        Lps = px**2 + py**2 + pz**2
        
        # Theta 3
        # =======    
        c3 = (Lps - l2**2 - l3**2) / (2*l2*l3)
        
        if c3 > (1 + tol) or c3 < -(1 + tol):
            return {'success': False}
        elif c3 > (1 - tol) or c3 < -(1 - tol):
            # almost 1 or -1 => fix it
            c3 = sign(c3)
        
        # point should be reachable
        # TODO just use arccos
        s3 = sqrt(1 - c3**2)
        
        t3_1 = arctan2( s3, c3)
        t3_2 = arctan2(-s3, c3)
        
        # Theta 2
        # =======
        Lxy = sqrt(px**2 + py**2) # TODO must be greater than zero (numerical error?)
        A = l2 + l3 * c3 # often used expression
        B = l3 * s3
        # positive sign for s3
        t2_1 = arctan2(
        pz * A - Lxy * B,
        Lxy * A + pz * B
        )
        t2_2 = arctan2(
        pz * A + Lxy * B,
        -Lxy * A + pz * B
        )
        # negative sign for s3
        t2_3 = arctan2(
        pz * A + Lxy * B,
        Lxy * A - pz * B
        )
        t2_4 = arctan2(
        pz * A - Lxy * B,
        -Lxy * A - pz * B
        )
        
        # Theta 1
        # =======
        t1_1 = arctan2(py, px)
        t1_2 = arctan2(-py, -px)
        
        # 4 solutions
        # =========
        q_sol = [
        [t1_1, t2_1, t3_1],
        [t1_1, t2_3, t3_2],
        [t1_2, t2_2, t3_1],
        [t1_2, t2_4, t3_2]
        ]
        
        return {'success': True, 'sol': q_sol}

class Arm2(Robot):
    """ Articulated arm with first link length is NOT zeros
    In addition the last frame is rotated to get the Z axis pointing out
    along the hypothetical 4 joint when adding a wrist"""
    def __init__(self, a1=1, a2=1, a3=1):
        super().__init__(
                [Link(DHLink(a1, pi/2, 0, 0), 'r', Shape(a1, 0.1, 0.1)),
                 Link(DHLink(a2, 0   , 0, 0), 'r', Shape(a2, 0.1, 0.1)),
                 Link(DHLink(a3, pi/2, 0, 0), 'r', Shape(a3, 0.1, 0.1))]
                )
    
    def ik(self, T):
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
        q1_pos = arctan2(y, x)
        q1_neg = arctan2(-y, -x)
        
        # Theta 3
        # =======    
        # q3 two options, elbow up and elbow down
        # return solutions between in interval (-pi, pi)
        den = 2*a2*a3
        num = a1**2 - a2**2 - a3**2 + x**2 + y**2 + z**2
        
        c1 = np.cos(q1_pos); s1 = np.sin(q1_pos)
        c3 = (num - 2*a1*s1*y - 2*a1*x*c1 ) / den
        
        if c3 > (1 + tol) or c3 < -(1 + tol):
            reachable_pos = False
        else:
            if c3 > (1 - tol) or c3 < -(1 - tol):
                # almost 1 or -1 => fix it
                c3 = sign(c3)
            
            s3 = sqrt(1 - c3**2)
            q3_pos_a = arctan2( s3, c3)
            q3_pos_b = arctan2(-s3, c3)
        
        c1 = np.cos(q1_neg); s1 = np.sin(q1_neg)
        c3 = (num - 2*a1*s1*y - 2*a1*x*c1 ) / den
        
        if c3 > (1 + tol) or c3 < -(1 + tol):
            reachable_neg = False
        else:     
            if c3 > (1 - tol) or c3 < -(1 - tol):
                # almost 1 or -1 => fix it
                c3 = sign(c3)
            #q3_a =  np.arccos(c3)
            #q3_b = -q3_a
            s3 = sqrt(1 - c3**2)
            q3_neg_a = arctan2( s3, c3)
            q3_neg_b = arctan2(-s3, c3)
        
        # Theta 2
        # =======
        if reachable_pos:
            s3 = np.sin(q3_pos_a)
            c3 = np.cos(q3_pos_b)
            L = np.sqrt(x**2 + y**2)
            
            q2_pos_a = np.arctan2((-a3*s3*(L - a1) + z*(a2 + a3*c3)),
                                  ( a3*s3*z + (L - a1)*(a2 + a3*c3)))
            q2_pos_b = np.arctan2(( a3*s3*(L - a1) + z*(a2 + a3*c3)),
                                  (-a3*s3*z + (L - a1)*(a2 + a3*c3)))
        if reachable_neg:
            s3 = np.sin(q3_neg_a)
            c3 = np.cos(q3_neg_b)
            q2_neg_a = np.arctan2(( a3*s3*(L + a1) + z*(a2 + a3*c3)),
                                  ( a3*s3*z - (L + a1)*(a2 + a3*c3)))
            q2_neg_b = np.arctan2((-a3*s3*(L + a1) + z*(a2 + a3*c3)),
                                  (-a3*s3*z - (L + a1)*(a2 + a3*c3)))
        #q2_neg_a = -q2_neg_a
        #q2_neg_b = -q2_neg_b
        
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
            return {'success': True, 'sol': q_sol}
        else:
            return {'success': False}

class Kuka(Robot):
    """ Robot combining AnthropomorphicArm and SphericalWrist
    """
    def __init__(self, a1=0.18, a2=0.6, d4=0.62, d6=0.115):
        # define kuka collision shapes
        s = [Shape(0.3, 0.2, 0.1),
           Shape(0.8, 0.2, 0.1),
           Shape(0.2, 0.1, 0.5),
           Shape(0.1, 0.2, 0.1),
           Shape(0.1, 0.1, 0.085),
           Shape(0.1, 0.1, 0.03)]
        # define transforms for collision shapes
        tfs = [pose_x(0, -0.09, 0  ,  0.05),
               pose_x(0, -0.3 , 0  , -0.05),
               pose_x(0,  0   , 0.05,  0.17),
               pose_x(0,  0   , 0.1 ,  0   ),
               pose_x(0,  0   , 0  ,  0.085/2),
               pose_x(0,  0   , 0  , -0.03/2)]
        
        # create robot
        super().__init__(
                [Link(DHLink(a1, pi/2, 0,   0), 'r', s[0], tf_shape=tfs[0]),
                 Link(DHLink(a2, 0   , 0,   0), 'r', s[1], tf_shape=tfs[1]),
                 Link(DHLink(0,  pi/2, 0,   0), 'r', s[2], tf_shape=tfs[2]),
                 Link(DHLink(0, -pi/2, d4,  0), 'r', s[3], tf_shape=tfs[3]),
                 Link(DHLink(0,  pi/2, 0,   0), 'r', s[4], tf_shape=tfs[4]),
                 Link(DHLink(0,  0   , d6,  0), 'r', s[5], tf_shape=tfs[5])]
                )
        self.arm = Arm2(a1=a1, a2=a2, a3=d4)
        self.wrist = SphericalWrist(d3=d6)
    
    def ik(self, T):
        # copy transform to change it without affecting the T given
        Tw = T.copy()
        # compensate for base
        Tw = np.dot(tf_inverse(self.tf_base), Tw)
        # compensate for tool frame
        if self.tool is not None:
            print('Kuka: adjusting for tool')
            Tw = np.dot(Tw, tf_inverse(self.tool.tf_tt))
        # compensate for d6, last link length
        d6 = self.links[5].dh.d
        v6 = np.dot(Tw[:3, :3], np.array([0, 0, d6]))
        Tw[:3, 3] = Tw[:3, 3] - v6
        sol_arm = self.arm.ik(Tw)
        if sol_arm['success']:
            solutions = []
            for q_arm in sol_arm['sol']:
                q_arm[2] = q_arm[2] + pi/2
                base_wrist = np.eye(4)
                # get orientation from arm fk
                base_wrist[:3, :3] = self.arm.fk(q_arm)[:3, :3]
                # position from given T (not used actually)
                base_wrist[3, :3] = Tw[3, :3]
                self.wrist.set_base_tf(base_wrist)
                sol_wrist = self.wrist.ik(Tw)
                if sol_wrist['success']:
                    for q_wrist in sol_wrist['sol']:
                        solutions.append(np.hstack((q_arm, q_wrist)))
            if len(solutions) > 0:
                return {'success': True, 'sol': solutions}
        return {'success': False}

class KukaOnRail(Robot):
    def __init__(self, a1=0.18, a2=0.6, d4=0.62, d6=0.115):        
        s  = [Shape(0.2, 0.2, 0.1),
               Shape(0.3, 0.2, 0.1),
               Shape(0.8, 0.2, 0.1),
               Shape(0.2, 0.1, 0.5),
               Shape(0.1, 0.2, 0.1),
               Shape(0.1, 0.1, 0.085),
               Shape(0.1, 0.1, 0.03)]
        
        tfs = [pose_x(0,  0   , 0  ,  -0.15),
               pose_x(0, -0.09, 0  ,  0.05),
               pose_x(0, -0.3 , 0  , -0.05),
               pose_x(0,  0   , 0.05,  0.17),
               pose_x(0,  0   , 0.1 ,  0   ),
               pose_x(0,  0   , 0  ,  0.085/2),
               pose_x(0,  0   , 0  , -0.03/2)]
        # create robot
        super().__init__(
                [Link(DHLink(0 , pi/2, 0,   0), 'p', s[0], tf_shape=tfs[0]),
                 Link(DHLink(a1, pi/2, 0,   0), 'r', s[1], tf_shape=tfs[1]),
                 Link(DHLink(a2, 0   , 0,   0), 'r', s[2], tf_shape=tfs[2]),
                 Link(DHLink(0,  pi/2, 0,   0), 'r', s[3], tf_shape=tfs[3]),
                 Link(DHLink(0, -pi/2, d4,  0), 'r', s[4], tf_shape=tfs[4]),
                 Link(DHLink(0,  pi/2, 0,   0), 'r', s[5], tf_shape=tfs[5]),
                 Link(DHLink(0,  0   , d6,  0), 'r', s[6], tf_shape=tfs[6])]
                )
        self.kuka = Kuka(a1=0.18, a2=0.6, d4=0.62, d6=0.115)
    
    def ik(self, T, q_fixed):
        # copy transform to change it without affecting the T given
        Tw = T.copy()
        # compensate for base
        Tw = np.dot(tf_inverse(self.tf_base), Tw)
        # compensate for tool frame
        if self.tool is not None:
            #print('Kuka: adjusting for tool')
            Tw = np.dot(Tw, tf_inverse(self.tool.tf_tt))
        
        # change base of helper robot according to fixed joint
        T1 = self.links[0].get_link_relative_transform(q_fixed)
        self.kuka.set_base_tf(T1)
        
        res = self.kuka.ik(Tw)
        if res['success']:
            q_sol = []
            for qi in res['sol']:
                q_sol.append(np.hstack((q_fixed, qi)))
            return {'success': True, 'sol': q_sol}
        else:
            return {'success': False}

class Puma(Robot):
    """ Puma parameters from https://github.com/uw-biorobotics/IKBT
    
    Note that I use the convention of Siciliano for the a index for
    a3 and a4

    ik code was in the old version, but the code was from someone else
    and I did not use it. TODO add link here.
    """
    def __init__(self, a3=0.432, a4=0.0203, d1=0.6, d3=0.1245, d4=0.432):
        super().__init__(
                [Link(DHLink(0 ,  0   , d1, 0), 'r', Shape(0.1, 0.1, 0.1)),
                 Link(DHLink(0 , -pi/2, 0 , 0), 'r', Shape(0.1, 0.1, 0.1)),
                 Link(DHLink(a3,  0   , d3, 0), 'r', Shape(0.1, 0.1, 0.1)),
                 Link(DHLink(a4, -pi/2, d4, 0), 'r', Shape(0.1, 0.1, 0.1)),
                 Link(DHLink(0 , -pi/2, 0 , 0), 'r', Shape(0.1, 0.1, 0.1)),
                 Link(DHLink(0 ,  pi/2, 0 , 0), 'r', Shape(0.1, 0.1, 0.1))]
                )
