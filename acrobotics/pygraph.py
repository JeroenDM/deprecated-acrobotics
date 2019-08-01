#!/usr/bin/env python3
"""
Python graph implementation + shortest path algorithms
Intial implementation from my ppr package (planar python robotics).

"""
import numpy as np
from queue import Queue
from pyquaternion import Quaternion


# def cost_function(node1, node2):
#     return np.sum(np.abs(node1.data - node2.data))


# def cost_function(node1, node2):
#     return Quaternion.distance(node1.data, node2.data)


def cost_function(node1, node2):
    return np.arccos(min(np.abs((node1.data @ node2.data)), 1.0))


class Node:
    def __init__(self, path_index, sample_index, data):
        self.path_index = path_index
        self.sample_index = sample_index
        self.data = data
        self.dist = np.inf
        self.parent = None
        self.visited = False

    def __str__(self):
        s = "Node (" + str(self.path_index) + ", "
        s += str(self.sample_index) + ") data: " + str(self.data)
        return s


class Graph:
    def __init__(self, data):
        self.data = data
        self.node_array = self.data_to_node_array()
        self.path_length = len(data)
        self.latest_total_cost = np.inf

    def data_to_node_array(self):
        na = []
        for i, data_col in enumerate(self.data):
            na.append([])
            for j, data_row in enumerate(data_col):
                na[-1].append(Node(i, j, data_row))
        return na

    def reset(self):
        for col in self.node_array:
            for node in col:
                node.dist = np.inf
                node.parent = None
                node.visited = False
        self.latest_total_cost = np.inf

    def get_neighbours(self, node):
        next_path_index = node.path_index + 1
        if next_path_index >= self.path_length:
            return []
        else:
            return self.node_array[next_path_index]

    def get_reachable_neighbours(self, node, max_cost=16):
        nb = self.get_neighbours(node)
        rnb = []
        for next_node in nb:
            if cost_function(node, next_node) < max_cost:
                rnb.append(next_node)
        return rnb

    def get_path(self, target_path_index=None):
        """ Return indices of shortest path nodes

        Returns [-1] when failed. This is to keep the type of the return object
        constant, for compatibility with c++ code later on.
        """
        # set default target path index if not specified
        if target_path_index is None:
            target_path_index = self.path_length - 1

        # look for the closest node in the target columns
        min_dist = np.inf
        closest_node = None
        for node in self.node_array[target_path_index]:
            if node.dist < min_dist:
                min_dist = node.dist
                closest_node = node

        if closest_node is None:
            return [-1]
        else:
            self.latest_total_cost = min_dist
            node_sample_index_list = []
            current_node = closest_node
            while current_node.parent is not None:
                node_sample_index_list.append(current_node.sample_index)
                current_node = current_node.parent

            node_sample_index_list.reverse()
            return node_sample_index_list

    def get_total_cost(self):
        return self.latest_total_cost

    def find_partial_path(self):
        # start at the back and look where the graph search got
        # before getting out of reachable neighbours
        current_path_index = self.path_length - 1
        found_shorter_path = False
        while current_path_index > 0 and not found_shorter_path:
            found_shorter_path = np.any(
                [n.parent is not None for n in self.node_array[current_path_index]]
            )
            current_path_index -= 1

        if found_shorter_path:
            return current_path_index + 1
        else:
            return -1

    def run_multi_source_bfs(self, max_cost=np.inf, start_path_index=0):
        # reset all node distances and visited status
        self.reset()

        Q = Queue()
        # add dummy before the first column
        # or the column specified by start_path_index
        # with distance zero to all these nodes
        dummy_node = Node(-1, 0, 0)
        for node in self.node_array[start_path_index]:
            node.parent = dummy_node
            node.visited = True
            node.dist = 0
            Q.put(node)

        # the dummy note is not in the queue
        # assume it is already handles and got us in a state
        # where all distances to the nodes in the first columns are 0
        # therefore these nodes are added to the queue in the above loop

        # now continue running the algorithm as usual
        while not Q.empty():
            current_node = Q.get()
            nb = self.get_reachable_neighbours(current_node, max_cost=max_cost)
            # nb is an empty list if their are no neighbours
            # TODO double cost function calculation

            for node in nb:
                new_dist = current_node.dist + cost_function(current_node, node)
                if new_dist < node.dist:
                    node.dist = new_dist
                    node.parent = current_node

                if not node.visited:
                    Q.put(node)
                    node.visited = True


def path_index_to_path(pi, data):
    res = []
    for i in range(len(pi)):
        qki = data[i][pi[i]]
        res.append(qki)
    return res


def get_shortest_path(data):
    g = Graph(data)
    g.run_multi_source_bfs()
    pi = g.get_path()
    cost = g.get_total_cost()
    if pi[0] == -1:
        return {"success": False}
    else:
        path = path_index_to_path(pi, data)
        return {"success": True, "path": path, "total_cost": cost, "single_path": True}


def get_shortest_partial_path(data, max_step_cost):
    g = Graph(data)
    g.run_multi_source_bfs(max_cost=max_step_cost)
    pi = g.get_path()
    cost = g.get_total_cost()
    if pi[0] == -1:
        split_index = g.find_partial_path()
        if split_index is -1:
            # TODO we could look for a path, excluding the first point
            return {"success": False, "info": "Stuck after first trajectory point"}

        # find the first segment up to the split index
        pi1 = g.get_path(target_path_index=split_index)
        cost1 = g.get_total_cost()
        path1 = path_index_to_path(pi1, data)

        return {"success": False, "path": path1, "total_cost": cost1, "i": split_index}
    else:
        # A complete path from start_path_index to finish is found
        path = path_index_to_path(pi, data)
        return {"success": True, "path": path, "total_cost": cost}


def get_shortest_path_segments(data, max_step_cost):
    # iteratively find shortest path in subset of data, start with all data
    current_data = data
    current_start_index = 0
    finished = False
    paths = []
    path_start_indices = []
    path_costs = []
    while not finished and current_start_index < (len(data) - 2):
        res = get_shortest_partial_path(current_data, max_step_cost)
        if res["success"]:
            finished = True
            paths.append(res["path"])
            path_start_indices.append(current_start_index)
            path_costs.append(res["total_cost"])
        else:
            if "i" in res:
                current_start_index = res["i"] + 1
                current_data = data[current_start_index:]
                paths.append(res["path"])
                path_start_indices.append(current_start_index)
                path_costs.append(res["total_cost"])
            else:
                # got stuck at the first trajectory point
                # res = {'success': False, 'info': 'Stuck after first trajectory point'}
                return res

    if finished:
        sol = {"success": True, "path": paths[0], "total_cost": path_costs[0]}

        # only one solution found
        if len(paths) == 1:
            sol["single_path"] = True
        # Path split in multiple segments
        else:
            sol["single_path"] = False
            sol["split_points"] = path_start_indices
            sol["extra_segments"] = []
            for i in range(1, len(paths)):
                si = {"path": paths[i], "total_cost": path_costs[i]}
                sol["extra_segments"].append(si)
    else:
        # TODO return more than only first part
        sol = {
            "success": False,
            "path": paths[0],
            "split_points": path_start_indices,
            "total_cost": path_costs[0],
            "info": "Path segments do not reach the end",
        }

    return sol
