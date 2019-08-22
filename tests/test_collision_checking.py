import numpy as np
from acrobotics.resources.robots import Kuka
from acrobotics.geometry import Shape, Scene
from acrobotics.util import pose_x

q1 = [-0.08307708, -0.61114407, 3.64637661, 0.77815741, -2.17606544, 2.35626125]

table = Shape(0.5, 0.5, 0.1)
table_tf = pose_x(0.0, 0.8, 0.0, 0.0)
scene1 = Scene([table], [table_tf])


class TestKukaCollisions:
    def test_scene_1(self):
        robot = Kuka()
        res = robot.is_in_collision(q1, scene1)
        assert res
