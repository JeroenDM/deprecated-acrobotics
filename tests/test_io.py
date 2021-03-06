import json
from acrobotics.io import *

test_data = """{
  "name" : "line_orient_free",
  "path" :
  {
    "type" : "LINE",
    "tolerance" : "orientation_free",
    "start_pose" :
    {
      "xyz" : [0.8, -0.2, 0.2],
      "rpy" : [0.0, 0.0, 0.0]
    },
    "end_point" : [0.8, 0.2, 0.2],
    "num_points" : 10
  },
  "obstacles" :
  [
    {
      "type" : "box",
      "size" : [0.5, 0.5, 0.1],
      "xyz" : [0.8, 0.0, 0.12],
      "rpy" : [0.0, 0.0, 0.0]
    }
  ]
}"""

# test_settings = """{"num_samples": 200, "graph_search_method": "dijkstra"}"""


def test_parse_task_data():
    task = parse_task_data(json.loads(test_data))
    assert len(task.path) == 10
