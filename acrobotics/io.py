"""
Fucntions to load tasks and other settings from json files.
"""
import json
import numpy as np

from .geometry import Shape, Collection
from .path import FreeOrientationPt
from .util import rot_x, rot_y, rot_z


def create_line(p1, p2, num_points):
    return np.linspace(np.array(p1), np.array(p2), num_points)


def create_orientation_free_path(pos):
    return [FreeOrientationPt(pi) for pi in pos]


def parse_path(d):
    pd = d["path"]

    if pd["type"] == "LINE":
        pos = create_line(pd["p1"], pd["p2"], pd["num_points"])
    else:
        msg = "Unkown path type: '{}'\n".format(pd["type"])
        msg += "Options are: {}\n".format(["LINE"])
        raise Exception(msg)

    if pd["tolerance"] == "orientation_free":
        path = create_orientation_free_path(pos)
    else:
        msg = "Unkown tolerance type: '{}'\n".format(pd["tolerance"])
        msg += "Options are: {}\n".format(["orientation_free"])
        raise Exception(msg)

    return path


def create_transform(xyz, rpy):
    tf = np.eye(4)
    tf[:3, 3] = np.array(xyz)
    tf[:3, :3] = rot_x(rpy[0]) @ rot_y(rpy[1]) @ rot_z(rpy[2])
    return tf


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

    return Collection(shapes, shape_transforms)


def parse_task_data(data):
    path = parse_path(data)
    scene = parse_obstacles(data)
    return path, scene


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
