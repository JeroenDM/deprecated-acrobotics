#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
from numpy import sin, cos, array, eye, vstack, pi, dot
from numpy.random import uniform, rand, seed

seed(41)  # make tests repeatable

from acrobotics.robot import Robot, DHLink, Link
from acrobotics.resources.robots import (
    PlanarArm,
    SphericalArm,
    AnthropomorphicArm,
    SphericalWrist,
    Arm2,
    Kuka,
    KukaOnRail,
)
from acrobotics.geometry import Shape, Scene
from acrobotics.util import pose_x
from .kuka_forward_kinematics import fk_kuka

from acrobotics.resources.torch_model import torch

# three link planar arm (p.69)
robot1 = PlanarArm(a1=1, a2=1.5, a3=0.5)

# spherical arm (p.72)
robot2 = SphericalArm(d2=2.6)

# fk_AnthropomorphicArm p 73
robot3 = AnthropomorphicArm()

robot4 = Arm2()


class TestForwardKinematics:
    # compare with analytic solution from the book
    # "Robotics: modelling, planning and control"
    def fk_PlanarArm(self, q, links):
        a1, a2, a3 = links[0].dh.a, links[1].dh.a, links[2].dh.a
        c123 = cos(q[0] + q[1] + q[2])
        s123 = sin(q[0] + q[1] + q[2])
        T = eye(4)
        T[0, 0] = c123
        T[0, 1] = -s123
        T[1, 0] = s123
        T[1, 1] = c123
        T[0, 3] = a1 * cos(q[0]) + a2 * cos(q[0] + q[1]) + a3 * c123
        T[1, 3] = a1 * sin(q[0]) + a2 * sin(q[0] + q[1]) + a3 * s123
        return T

    def fk_SphericalArm(self, q, links):
        d2, d3 = links[1].dh.d, q[2]
        c1 = cos(q[0])
        s1 = sin(q[0])
        c2 = cos(q[1])
        s2 = sin(q[1])
        T = eye(4)
        T[0, 0:3] = array([c1 * c2, -s1, c1 * s2])
        T[0, 3] = c1 * s2 * d3 - s1 * d2
        T[1, 0:3] = array([s1 * c2, c1, s1 * s2])
        T[1, 3] = s1 * s2 * d3 + c1 * d2
        T[2, 0:3] = array([-s2, 0, c2])
        T[2, 3] = c2 * d3
        return T

    def fk_AnthropomorphicArm(self, q, links):
        a2, a3 = links[1].dh.a, links[2].dh.a
        c1 = cos(q[0])
        s1 = sin(q[0])
        c2 = cos(q[1])
        s2 = sin(q[1])
        c23 = cos(q[1] + q[2])
        s23 = sin(q[1] + q[2])
        T = eye(4)
        T[0, 0:3] = array([c1 * c23, -c1 * s23, s1])
        T[0, 3] = c1 * (a2 * c2 + a3 * c23)
        T[1, 0:3] = array([s1 * c23, -s1 * s23, -c1])
        T[1, 3] = s1 * (a2 * c2 + a3 * c23)
        T[2, 0:3] = array([s23, c23, 0])
        T[2, 3] = a2 * s2 + a3 * s23
        return T

    def fk_Arm2(self, q, links):
        a1, a2, a3 = links[0].dh.a, links[1].dh.a, links[2].dh.a
        c1 = cos(q[0])
        s1 = sin(q[0])
        c2 = cos(q[1])
        s2 = sin(q[1])
        c23 = cos(q[1] + q[2])
        s23 = sin(q[1] + q[2])
        T = eye(4)
        T[0, 0:3] = array([c1 * c23, s1, c1 * s23])
        T[1, 0:3] = array([s1 * c23, -c1, s1 * s23])
        T[2, 0:3] = array([s23, 0, -c23])
        T[0, 3] = c1 * (a1 + a2 * c2 + a3 * c23)
        T[1, 3] = s1 * (a1 + a2 * c2 + a3 * c23)
        T[2, 3] = a2 * s2 + a3 * s23
        return T

    def generate_random_configurations(self, robot, N=5):
        C = []
        for jl in robot.joint_limits:
            C.append(uniform(jl.lower, jl.upper, size=N))
        return vstack(C).T

    def test_PlanarArm_robot(self):
        q_test = self.generate_random_configurations(robot1)
        for qi in q_test:
            Tactual = robot1.fk(qi)
            Tdesired = self.fk_PlanarArm(qi, robot1.links)
            assert_almost_equal(Tactual, Tdesired)

    def test_SphericalArm_robot(self):
        q_test = self.generate_random_configurations(robot2)
        for qi in q_test:
            Tactual = robot2.fk(qi)
            Tdesired = self.fk_SphericalArm(qi, robot2.links)
            assert_almost_equal(Tactual, Tdesired)

    def test_AnthropomorphicArm_robot(self):
        q_test = self.generate_random_configurations(robot3)
        for qi in q_test:
            Tactual = robot3.fk(qi)
            Tdesired = self.fk_AnthropomorphicArm(qi, robot3.links)
            assert_almost_equal(Tactual, Tdesired)

    def test_Arm2_robot(self):
        q_test = self.generate_random_configurations(robot4)
        for qi in q_test:
            Tactual = robot4.fk(qi)
            Tdesired = self.fk_Arm2(qi, robot4.links)
            assert_almost_equal(Tactual, Tdesired)

    def test_Arm2_tool_robot(self):
        bot = Arm2()
        bot.tool = torch
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = self.fk_Arm2(qi, bot.links)
            Tdesired = dot(Tdesired, torch.tf_tt)
            assert_almost_equal(Tactual, Tdesired)

    def test_Kuka_robot(self):
        bot = Kuka()
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fk_kuka(qi)
            assert_almost_equal(Tactual, Tdesired)

    def test_Kuka_tool_robot(self):
        bot = Kuka()
        bot.tool = torch
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fk_kuka(qi)
            Tdesired = dot(Tdesired, torch.tf_tt)
            assert_almost_equal(Tactual, Tdesired)

    def test_Kuka_base_robot(self):
        bot = Kuka()
        bot.tf_base = pose_x(0.5, 0.1, 0.2, 0.3)
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fk_kuka(qi)
            Tdesired = dot(bot.tf_base, Tdesired)
            assert_almost_equal(Tactual, Tdesired)

    def test_Kuka_on_rail_robot(self):
        bot = KukaOnRail()
        q_test = self.generate_random_configurations(bot)
        for qi in q_test:
            Tactual = bot.fk(qi)
            Tdesired = fk_kuka(qi[1:])
            # Ti = bot.links[0].get_link_relative_transform(qi[0])
            # Tdesired = np.dot(Ti, Tdesired)
            Tdesired = np.dot(pose_x(pi / 2, 0, 0, qi[0]), Tdesired)
            assert_almost_equal(Tactual, Tdesired)

    def test_PlanarArm_base(self):
        robot1.tf_base = pose_x(1.5, 0.3, 0.5, 1.2)
        q_test = self.generate_random_configurations(robot1)
        for qi in q_test:
            Tactual = robot1.fk(qi)
            Tdesired = self.fk_PlanarArm(qi, robot1.links)
            Tdesired = dot(robot1.tf_base, Tdesired)
            assert_almost_equal(Tactual, Tdesired)


#    def test_sw_base(self):
#
#        bot = SphericalWrist()
#        bot.tf_base = pose_x(1.5, 0.3, 0.5, 1.2)
#        q_test = self.generate_random_configurations(bot)
#        for qi in q_test:
#            Tactual = bot.fk(qi)
#            Tdesired = self.fk_PlanarArm(qi, robot1.links)
#            Tdesired = dot(tf_base, Tdesired)
#            assert_almost_equal(Tactual, Tdesired)


class TestCollisionChecking:
    def test_kuka_self_collision(self):
        bot = Kuka()
        gl = [l.geometry for l in bot.links]
        q0 = [0, pi / 2, 0, 0, 0, 0]
        q_self_collision = [0, 1.5, -1.3, 0, -1.5, 0]
        tf1 = bot.fk_all_links(q0)
        a1 = bot._check_self_collision(tf1, gl)
        assert a1 is False
        tf2 = bot.fk_all_links(q_self_collision)
        a2 = bot._check_self_collision(tf2, gl)
        assert a2 is True

    def test_kuka_collision(self):
        bot = Kuka()
        q0 = [0, pi / 2, 0, 0, 0, 0]
        obj1 = Scene(
            [Shape(0.2, 0.3, 0.5), Shape(0.1, 0.3, 0.1)],
            [pose_x(0, 0.75, 0, 0.5), pose_x(0, 0.75, 0.5, 0.5)],
        )
        obj2 = Scene(
            [Shape(0.2, 0.3, 0.5), Shape(0.1, 0.3, 0.1)],
            [pose_x(0, 0.3, -0.7, 0.5), pose_x(0, 0.75, 0.5, 0.5)],
        )
        a1 = bot.is_in_collision(q0, obj1)
        assert a1 is True
        a2 = bot.is_in_collision(q0, obj2)
        assert a2 is False


class TestIK:
    def test_aa_random(self):
        bot = AnthropomorphicArm()
        N = 20
        q_rand = rand(N, 3) * 2 * pi - pi
        for qi in q_rand:
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            for q_sol in resi.solutions:
                p2 = bot.fk(q_sol)[:3, 3]
                assert_almost_equal(T1[:3, 3], p2)

    def test_sw_random(self):
        bot = SphericalWrist()
        N = 20
        q_rand = rand(N, 3) * 2 * pi - pi
        for qi in q_rand:
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            for q_sol in resi.solutions:
                R2 = bot.fk(q_sol)[:3, :3]
                assert_almost_equal(T1[:3, :3], R2)

    def test_sw_random_other_base(self):
        bot = SphericalWrist()
        bot.tf_base = pose_x(1.5, 0.3, 0.5, 1.2)
        N = 20
        q_rand = rand(N, 3) * 2 * pi - pi
        for qi in q_rand:
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            for q_sol in resi.solutions:
                R2 = bot.fk(q_sol)[:3, :3]
                assert_almost_equal(T1[:3, :3], R2)

    def test_arm2_random(self):
        bot = Arm2()
        N = 20
        q_rand = rand(N, 3) * 2 * pi - pi
        for qi in q_rand:
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            if resi.success:
                for q_sol in resi.solutions:
                    p2 = bot.fk(q_sol)[:3, 3]
                    assert_almost_equal(T1[:3, 3], p2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    #    def test_arm2_tool_random(self):
    #        bot = Arm2()
    #        bot.tf_tool = pose_x(0, 0.1, 0, 0)
    #        N = 20
    #        q_rand = rand(N, 3) * 2 * pi - pi
    #        for qi in q_rand:
    #            T1 = bot.fk(qi)
    #            resi = bot.ik(T1)
    #            if resi['success']:
    #                for q_sol in resi['sol']:
    #                    p2 = bot.fk(q_sol)[:3, 3]
    #                    assert_almost_equal(T1[:3, 3], p2)
    #            else:
    #                # somethings is wrong, should be reachable
    #                print(resi)
    #                assert_almost_equal(qi, 0)

    def test_kuka_random(self):
        bot = Kuka()
        N = 20
        q_rand = rand(N, 6) * 2 * pi - pi
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    def test_kuka_tool_random(self):
        bot = Kuka()
        bot.tool = torch
        N = 20
        q_rand = rand(N, 6) * 2 * pi - pi
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    def test_kuka_base_random(self):
        bot = Kuka()
        bot.tf_base = pose_x(0.1, 0.02, 0.01, -0.01)
        N = 20
        q_rand = rand(N, 6) * 2 * pi - pi
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1)
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    def test_kuka_on_rail_random(self):
        bot = KukaOnRail()
        N = 20
        q_rand = rand(N, 7) * 2 * pi - pi
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1, qi[0])
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)

    def test_kuka_on_rail_tool_random(self):
        bot = KukaOnRail()
        bot.tool = torch
        N = 20
        q_rand = rand(N, 7) * 2 * pi - pi
        for qi in q_rand:
            print(qi)
            T1 = bot.fk(qi)
            resi = bot.ik(T1, qi[0])
            if resi.success:
                for q_sol in resi.solutions:
                    print(q_sol)
                    T2 = bot.fk(q_sol)
                    assert_almost_equal(T1, T2)
            else:
                # somethings is wrong, should be reachable
                print(resi)
                assert_almost_equal(qi, 0)
