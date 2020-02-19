import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

from acrobotics.util import plot_reference_frame
from acrobotics.path import point_to_frame
from acrobotics.resources.workpiece_model import workpiece

from acrobotics.planning import (
    cart_to_joint_simple,
    get_shortest_path,
    cart_to_joint_no_redundancy,
)

from acrobotics.urdfio import import_urdf
from acrobotics.io import import_paths


def create_layout_slots(rows, cols, margin):
    slot_h = 1.0 / rows
    slot_w = 1.0 / cols

    slots = []
    for row in range(rows):
        slots.append([])
        for col in range(cols):
            left, bottom = col * slot_w + margin, row * slot_h + margin
            width, height = slot_w - 2 * margin, slot_h - 2 * margin
            slots[-1].append([left, bottom, width, height])
    return slots


def add_button(fig, label, rect):
    ax = fig.add_axes(rect)
    btn = Button(ax, label)
    return {"ax": ax, "wd": btn}


def add_textbox(fig, label, rect):
    ax = fig.add_axes(rect)
    box = TextBox(ax, label)
    return {"ax": ax, "wd": box}


def create_layout():
    slots = create_layout_slots(10, 4, 0.02)
    widgets = {}

    fig = plt.figure(figsize=(10, 7))

    ax_view = fig.add_axes([0.5, 0, 0.5, 1.0], projection="3d")
    xlim = [-1, 1]
    ylim = [-1, 1]
    zlim = [-1, 1]
    ax_view.set_xlim3d(xlim)
    ax_view.set_ylim3d(ylim)
    ax_view.set_zlim3d(zlim)
    ax_view.set_xlabel("X")
    ax_view.set_ylabel("Y")
    ax_view.set_zlabel("Z")

    widgets["scene_file"] = add_textbox(fig, "Scene urdf", slots[9][0])
    widgets["load_scene"] = add_button(fig, "Load scene", slots[8][0])

    widgets["paths_file"] = add_textbox(fig, "Paths file", slots[7][0])
    widgets["load_paths"] = add_button(fig, "Load paths", slots[6][0])

    widgets["plan_btn"] = add_button(fig, "Plan", slots[3][0])
    widgets["show_path"] = add_button(fig, "Show solution", slots[3][1])

    ax_log = fig.add_axes(slots[2][0])
    log_text = ax_log.text(
        0.05, 0.95, "", transform=ax_log.transAxes, verticalalignment="top",
    )
    ax_log.set_axis_off()
    # ax_log.xaxis.set_ticklabels([])
    # ax_log.xaxis.set_ticks([])
    # ax_log.yaxis.set_ticklabels([])
    # ax_log.yaxis.set_ticks([])
    widgets["log_text"] = {"ax": ax_log, "wd": log_text}

    return fig, ax_view, widgets


class TaskEditor:
    """ A simple gui to load and interact with tasks and robots in 3D. """

    axcolor = "lightgoldenrodyellow"
    default_scene = "scenes/path_in_box.urdf"
    default_path = "tasks/path_in_box.json"
    # layout utils

    def __init__(self, robot, wdir):

        self.robot = robot

        self.scene = None
        self.paths = []
        self.path = []
        self.wdir = wdir
        self.filename = self.default_scene
        self.paths_filename = self.default_path

        # setup
        self.fig, self.ax, self.widgets = create_layout()
        self.bind_callbacks()

        # self.plot_setup()
        self.log_text = ""

        # vars
        self.current_solution = None

    def log(self, message):
        self.log_text += "\n" + str(message)
        self.widgets["log_text"]["wd"].set_text(self.log_text)
        self.fig.canvas.draw_idle()

    def submit_file_name(self, name):
        self.filename = name

    def submit_path_file_name(self, name):
        self.paths_filename = name

    def load_scene(self, event):
        self.scene = import_urdf(self.wdir + "/" + self.filename)
        self.plot_setup()

    def load_paths(self, event):
        self.paths = import_paths(self.wdir + "/" + self.paths_filename)
        self.path = self.paths[0]
        self.plot_setup()

    def plot_setup(self):
        q0 = [0, 0, 1.5, 0, 0, 0, 0]
        self.robot.plot(self.ax, q0, c="k")
        self.scene.plot(self.ax, c="g")
        for path in self.paths:
            for tp in path:
                tf = point_to_frame(tp.p_nominal)
                plot_reference_frame(self.ax, tf)
        self.fig.canvas.draw_idle()

    def bind_callbacks(self):
        self.widgets["plan_btn"]["wd"].on_clicked(lambda event: self.plan(event))
        self.widgets["show_path"]["wd"].on_clicked(lambda event: self.show_path(event))
        self.widgets["load_scene"]["wd"].on_clicked(
            lambda event: self.load_scene(event)
        )
        self.widgets["scene_file"]["wd"].on_submit(
            lambda name: self.submit_file_name(name)
        )
        self.widgets["load_paths"]["wd"].on_clicked(
            lambda event: self.load_paths(event)
        )
        self.widgets["paths_file"]["wd"].on_submit(
            lambda name: self.submit_path_file_name(name)
        )

    def plan(self, event):
        self.log("Planning...")

        if self.robot.ndof > 6:
            qf_samples = np.linspace(-0.5, 0.5, 10)
            Q = cart_to_joint_simple(self.robot, self.path, self.scene, qf_samples)
        else:
            Q = cart_to_joint_no_redundancy(self.robot, self.path, self.scene)

        self.log([len(qi) for qi in Q])

        w = np.array([1, 1, 1, 1, 1, 1, 1], dtype="float32")
        res = get_shortest_path(Q, method="dijkstra", weights=w)

        self.log(res["success"])
        qp_sol = res["path"]
        self.current_solution = qp_sol

    def show_path(self, event):
        if self.current_solution is not None:
            self.ax.clear()
            self.plot_setup()
            self.robot.animate_path(self.fig, self.ax, self.current_solution)
        else:
            self.log("No recent path to show.")
