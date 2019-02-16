import numpy as np
from numpy.testing import assert_almost_equal
from acrobotics.cpp.graph import Graph

data1 = np.array([[0, 0], [0, 1]], dtype='float32')
data2 = np.array([[1, -1], [1, 0], [1, 1]], dtype='float32')
data3 = np.array([[0, 2], [2, 2]], dtype='float32')

class TestGraph:
  def test_init(self):
    g = Graph()
  
  def test_add_data(self):
    g = Graph()
    g.add_data_column(data1)
  
  def test_shortest_path(self):
    g = Graph()

    g.add_data_column(data1)
    g.add_data_column(data2)
    g.add_data_column(data3)

    g.init()
    g.run_dijkstra()
    p = g.get_path(3)
    assert_almost_equal(p, np.array([1, 2, 0]))
