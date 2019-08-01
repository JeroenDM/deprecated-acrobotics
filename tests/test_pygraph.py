# #!/usr/bin/env python3
# from acrobotics.pygraph import Node, Graph, cost_function
# from acrobotics.pygraph import get_shortest_path, get_shortest_path_segments
#
# import numpy as np
# from numpy.testing import assert_almost_equal
#
# data1 = [
#     np.array([[0, 0]]),
#     np.array([[1, -1], [1, 0], [1, 1]]),
#     np.array([[0, 2], [2, 2]]),
#     np.array([[4, 5], [5, 9]]),
#     np.array([[4, 6], [5, 10]]),
#     np.array([[4, 13], [5, 11]]),
# ]
#
# data2 = [np.array([[0, 0]]), np.array([[-1, -1], [1, 1]]), np.array([[-2, -2], [2, 2]])]
#
# data3 = [np.array([[0, 0]]), np.array([]), np.array([[-2, -2], [2, 2]])]
#
# data4 = [
#     np.array([[0, 0]]),
#     np.array([[1, -1], [1, 0], [1, 1]]),
#     np.array([[0, 2], [2, 2]]),
#     np.array([[4, 5], [5, 9]]),
#     np.array([[4, 6], [5, 10]]),
#     np.array([[99, 99], [99, 99]]),
# ]
#
#
# class TestNode:
#     def test_init(self):
#         data = np.array([1, 2, 3])
#         n1 = Node(0, 1, data)
#
#         # check default values
#         assert n1.parent == None
#         assert n1.dist == np.inf
#         assert n1.visited == False
#
#     def test_print_string(self):
#         n1 = Node(0, 1, np.array([1, 2, 3]))
#         a = n1.__str__()
#         assert a == "Node (0, 1) data: [1 2 3]"
#
#
# def test_cost_function():
#     n1 = Node(0, 1, np.array([1, 2, 3]))
#     n2 = Node(0, 1, np.array([-1, 2, 4]))
#     a = cost_function(n1, n2)
#     assert a == 3
#
#
# class TestGraph:
#     def test_init(self):
#         g = Graph(data1)
#         assert g.path_length == len(data1)
#
#     def test_data_to_node_array(self):
#         g = Graph(data2)
#         na = g.node_array
#         for i in range(len(data2)):
#             for j in range(len(data2[i])):
#                 assert_almost_equal(na[i][j].data, data2[i][j])
#
#
# def test_get_shortest_path():
#     res1 = get_shortest_path(data1)
#     assert res1["success"] == True
#     assert res1["total_cost"] == 16
#     assert res1["single_path"] == True
#     d = np.array([[0, 0], [1, 0], [2, 2], [4, 5], [4, 6], [5, 11]])
#     assert_almost_equal(res1["path"], d)
#
#
# def test_get_shortest_path_failed():
#     res1 = get_shortest_path(data3)
#     assert res1["success"] == False
#
#
# class TestShortestPathSegments:
#     def test_two_segments(self):
#         res1 = get_shortest_path_segments(data1, 5)
#         assert res1["success"] == True
#         assert res1["total_cost"] == 4
#         assert res1["single_path"] == False
#         d = np.array([[0, 0], [1, 0], [0, 2]])
#         assert_almost_equal(res1["path"], d)
#         assert res1["split_points"] == [3, 3]
#
#         subres = res1["extra_segments"][0]
#         assert subres["total_cost"] == 2
#         d2 = np.array([[5, 9], [5, 10], [5, 11]])
#         assert_almost_equal(subres["path"], d2)
#
#     def test_failed(self):
#         res1 = get_shortest_path_segments(data1, 2)
#         assert res1["success"] == False
#         assert res1["info"] == "Stuck after first trajectory point"
#
#     def test_single_path(self):
#         res1 = get_shortest_path_segments(data1, 20)
#         assert res1["success"] == True
#         assert res1["total_cost"] == 16
#         assert res1["single_path"] == True
#         d = np.array([[0, 0], [1, 0], [2, 2], [4, 5], [4, 6], [5, 11]])
#         assert_almost_equal(res1["path"], d)
#
#     def test_partly_failed(self):
#         res1 = get_shortest_path_segments(data4, 20)
#         assert res1["success"] == False
#         assert res1["total_cost"] == 10
#         d = np.array([[0, 0], [1, 0], [2, 2], [4, 5], [4, 6]])
#         assert_almost_equal(res1["path"], d)
