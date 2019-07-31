import numpy as np
from pyquaternion import Quaternion

from acrobotics.pygraph import Graph, get_shortest_path
from acrobotics.util import sample_SO3

data = []
for i in range(5):
    data.append(sample_SO3(10, rep="quat"))

# g = Graph(data)
#
# g.run_multi_source_bfs()
#
# pi = g.get_path()
#
# cost = g.get_total_cost()
#
# print(pi)
# print(cost)

sol = get_shortest_path(data)
print(sol)
