"""
Fucntions to load tasks and other settings from json files.
"""
import json
import numpy as np
from .pyquat_extended import QuaternionExtended as Quaternion
from numpy.linalg import norm

from .geometry import Shape, Scene
from .path.path_pt import FreeOrientationPt, TolPositionPt
from .path.toleranced_number import TolerancedNumber
from .util import rot_x, rot_y, rot_z
from .planning import PlanningTask


def create_transform(xyz, rpy):
    tf = np.eye(4)
    tf[:3, 3] = np.array(xyz)
    tf[:3, :3] = rot_x(rpy[0]) @ rot_y(rpy[1]) @ rot_z(rpy[2])
    return tf


def rpy_to_quaternion(rpy):
    R = rot_x(rpy[0]) @ rot_y(rpy[1]) @ rot_z(rpy[2])
    return Quaternion(matrix=R)


def create_line(start_pose, end_point, num_points):
    return np.linspace(np.array(start_pose["xyz"]), np.array(end_point), num_points)


def create_circle(mid, start, axis, num_points):
    e = axis / norm(axis)
    c = np.array(mid)
    v = np.array(start) - c
    angle = np.linspace(0, 2 * np.pi, num_points)
    pos = []
    for a in angle:
        v_rot = np.cos(a) * v + np.sin(a) * np.cross(e, v)
        v_rot = v_rot + (1 - np.cos(a)) * (e @ v) * e
        pos.append(c + v_rot)
    return np.array(pos)


def create_orientation_free_path(pos):
    return [FreeOrientationPt(pi) for pi in pos]


def create_symmetric_position_tolerance(pos, rpy, pos_tol):
    q = rpy_to_quaternion(rpy)
    path = []
    for p_nominal in pos:
        xyz = []
        for pi, tol in zip(p_nominal, pos_tol):
            if tol > 0.0:
                xyz.append(TolerancedNumber(pi - tol, pi + tol))
            else:
                xyz.append(pi)
        path.append(TolPositionPt(xyz, q))
    return path


def parse_path(d):
    pd = d["path"]

    if pd["type"] == "LINE":
        pos = create_line(pd["start_pose"], pd["end_point"], pd["num_points"])
    elif pd["type"] == "CIRC":
        pos = create_circle(pd["mid"], pd["start"], pd["axis"], pd["num_points"])
    else:
        msg = "Unkown path type: '{}'\n".format(pd["type"])
        msg += "Options are: {}\n".format(["LINE", "CIRC"])
        raise Exception(msg)

    if pd["tolerance"] == "orientation_free":
        path = create_orientation_free_path(pos)
    elif pd["tolerance"] == "position_tolerance":
        path = create_symmetric_position_tolerance(
            pos, pd["start_pose"]["rpy"], pd["xyz_tol"]
        )
    else:
        msg = "Unkown tolerance type: '{}'\n".format(pd["tolerance"])
        msg += "Options are: {}\n".format(["orientation_free"])
        raise Exception(msg)

    return path


def create_box(d):
    box = Shape(*d["size"])
    box_transform = create_transform(d["xyz"], d["rpy"])
    return box, box_transform


def parse_shape(shape_dict):
    if shape_dict["type"] == "box":
        return create_box(shape_dict)
    else:
        raise Exception("Invalid shape type: {}".format(shape_dict["type"]))


def parse_obstacles(d):
    od = d["obstacles"]
    shapes = []
    shape_transforms = []
    for shape_dict in od:
        s, tf = parse_shape(shape_dict)
        shapes.append(s)
        shape_transforms.append(tf)

    return Scene(shapes, shape_transforms)


def parse_task_data(data):
    path = parse_path(data)
    scene = parse_obstacles(data)
    return PlanningTask(path, scene)


def load_task(filepath):
    with open(filepath) as file:
        data = json.load(file)
    return parse_task_data(data)


def load_settings(filepath):
    with open(filepath) as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    p, s = load_task("examples/line_orient_free.json")
    for pi in p:
        print(pi)
    print(s)
