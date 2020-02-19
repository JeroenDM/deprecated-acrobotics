import numpy as np
from string import Template


from acrobotics.resources.workpiece_model import workpiece
from acrobotics.urdfio import export_urdf, import_urdf
from acrobotics.io import load_task

# scene = import_urdf(
#     "/home/jeroen/Documents/github/deprecated-acrobotics/examples/halfopen_box.urdf"
# )


task = load_task(
    "/home/jeroen/Documents/github/deprecated-acrobotics/examples/small_passage_2.json"
)

export_urdf(
    task.scene,
    "small_passage",
    "/home/jeroen/Documents/github/deprecated-acrobotics/examples",
)

import matplotlib.pyplot as plt
from acrobotics.util import get_default_axes3d

fig, ax = get_default_axes3d()
task.scene.plot(ax, c="g")
plt.show()
