#include <iostream>
#include <algorithm>
#include <math.h>
#include <queue>
#include "graph.h"

// dummy node to initialize parent nodes
// TODO accessing this nodes parent can lead to segmentation faults
Node DUMMY_NODE;

// sort function for dijkstra unvisited list
bool sort_function(Node *n1, Node *n2)
{
    return (*n2).dist < (*n1).dist;
}

// ===========================================================
// Graph search algorithms
// ===========================================================

void Graph::multi_source_bfs()
{
    std::queue<Node *> Q;
    Node *current;
    std::vector<Node *> neighbors;

    // add dummy parent node before all start nodes
    // with distance zero to all the start nodes
    for (Node &sn : na[0])
    {
        sn.dist = 0;
        sn.parent = &DUMMY_NODE;
        sn.visited = true;
        Q.push(&sn);
    }

    std::cout << "dummy colums setup" << std::endl;
    //print_node(DUMMY_NODE);
    //print_nodes(na[0]);

    while (!Q.empty())
    {
        // get the next node in line
        current = Q.front();
        Q.pop();

        neighbors = get_neighbors(current);
        for (Node *nb : neighbors)
        {
            // update neighbors distance
            float dist = (*current).dist + cost_function(*nb, *current);
            if (dist < (*nb).dist)
            {
                (*nb).dist = dist;
                (*nb).parent = current;
            }
            // add to queue if not yet visited
            if (!(*nb).visited)
            {
                Q.push(nb);
                (*nb).visited = true;
            }
        }
    }

    path_found = true;
}

void Graph::multi_source_dijkstra()
{
    //std::queue<Node*> Q;
    std::priority_queue<Node *, std::vector<Node *>, dijkstraSortFunction> Q;
    Node *current;
    std::vector<Node *> neighbors;

    // add dummy parent node before all start nodes
    // with distance zero to all the start nodes
    for (Node &sn : na[0])
    {
        sn.dist = 0;
        sn.parent = &DUMMY_NODE;
        sn.visited = true;
        Q.push(&sn);
    }

    std::cout << "dummy colums setup" << std::endl;
    //print_node(DUMMY_NODE);
    //print_nodes(na[0]);

    while (!Q.empty() and (num_goals_to_visit > 0))
    {
        // get the next node in line
        //current = Q.front();
        current = Q.top();
        Q.pop();

        // count the number of goal nodes that are reached
        if ((*current).path_index == max_path_index)
        {
            num_goals_to_visit--;
        }

        neighbors = get_neighbors(current);
        for (Node *nb : neighbors)
        {
            // update neighbors distance
            float dist = (*current).dist + cost_function(*nb, *current);
            if (dist < (*nb).dist)
            {
                (*nb).dist = dist;
                (*nb).parent = current;
            }
            // add to queue if not yet visited
            if (!(*nb).visited)
            {
                Q.push(nb);
                (*nb).visited = true;
            }
        }
    }

    path_found = true;
}

// ===========================================================
// private graph fucntions
// ===========================================================
void Graph::graph_data_to_node_array()
{
    for (std::size_t i = 0; i < gd.size(); ++i)
    {
        std::vector<Node> new_column;
        na.push_back(new_column);
        for (std::size_t j = 0; j < gd[i].size(); ++j)
        {
            Node new_node = {i, j, &gd[i][j], INF, &DUMMY_NODE, false};
            na[i].push_back(new_node);
        }
    }
}

void Graph::reset_node_array()
{
    for (std::size_t i = 0; i < na.size(); ++i)
    {
        for (std::size_t j = 0; j < na[i].size(); ++j)
        {
            na[i][j].dist = INF;
            na[i][j].parent = &DUMMY_NODE;
            na[i][j].visited = false;
        }
    }
}

void Graph::init_unvisited(std::vector<Node *> &uv)
{
    // do not add fist column (start at i=1)
    // start node in first column cannot reached the other nodes there
    for (std::size_t i = 1; i < na.size(); ++i)
    {
        for (Node &n : na[i])
        {
            uv.push_back(&n);
        }
    }
}

float Graph::cost_function(Node n1, Node n2)
{
    float cost = 0;
    for (std::size_t i = 0; i < (*n1.jv).size(); ++i)
    {
        cost += fabs((*n1.jv)[i] - (*n2.jv)[i]);
    }
    return cost;
}

void Graph::visit(Node *n)
{
    std::vector<Node *> neighbors;
    neighbors = get_neighbors(n);
    for (Node *nb : neighbors)
    {
        float dist = (*n).dist + cost_function(*nb, *n);
        if (dist < (*nb).dist)
        {
            (*nb).dist = dist;
            (*nb).parent = n;
        }
    }
}

std::vector<Node *> Graph::get_neighbors(Node *node)
{
    std::vector<Node *> result;
    // check if we are at the end of the graph
    if ((*node).path_index == (na.size() - 1))
    {
        return result; // empty vector
    }
    else
    {
        for (Node &n : na[(*node).path_index + 1])
        {
            result.push_back(&n);
        }
        return result;
    }
}

std::vector<Node *> Graph::get_path_nodes()
{
    std::vector<Node *> path;

    if (path_found)
    {
        // find last node with shortest distance to start
        float min_dist = INF;
        Node* goal = nullptr;
        for (auto &n : na.back())
        {
            if (n.dist < min_dist)
            {
                goal = &n;
                min_dist = n.dist;
            }
        }

        shortest_path_cost = min_dist;

        Node *current_node = goal;
        while ((*current_node).path_index > 0)
        {
            path.push_back(current_node);
            current_node = (*current_node).parent;
        }
        path.push_back(current_node);
        std::reverse(path.begin(), path.end());
        return path;
    }
    else
    {
        std::cout << "No path found" << std::endl;
        return path;
    }
}

float Graph::get_path_cost(std::vector<Node *> &path)
{
    float cost = 0.0;
    for (std::size_t i = 0; i < path.size() - 1; ++i)
    {
        cost += cost_function(*path[i], *path[i + 1]);
    }
    return cost;
}

// ===========================================================
// puplic graph fucntions
// ===========================================================
// rows containing the different solutions
// columns different joint values
void Graph::add_data_column(float *mat, int nrows, int ncols)
{
    int index;
    std::vector<joint_value> new_col;
    for (int i = 0; i < nrows; ++i)
    {
        joint_value new_jv;
        for (int j = 0; j < ncols; ++j)
        {
            index = j + ncols * i;
            new_jv.push_back(mat[index]);
        }
        new_col.push_back(new_jv);
    }
    gd.push_back(new_col);
}

void Graph::init()
{
    std::cout << "Initializing graph for planning" << std::endl;
    graph_data_to_node_array();
    max_path_index = na.size() - 1;
    num_goals_to_visit = na[max_path_index].size();
    std::cout << "Found " << num_goals_to_visit << " goal nodes." << std::endl;
    std::cout << "Index last path point " << max_path_index << std::endl;
    path_found = false;
}

void Graph::reset()
{
    reset_node_array();
    max_path_index = na.size() - 1;
    num_goals_to_visit = na[max_path_index].size();
}

void Graph::run_bfs()
{
    std::cout << "Running Breath-first Search " << std::endl;
    multi_source_bfs();
    std::vector<Node *> path = get_path_nodes();
    float cost = get_path_cost(path);
    std::cout << "Path found with cost: " << cost << std::endl;
}

void Graph::run_dijkstra()
{
    std::cout << "Running Dijkstra's algorithm " << std::endl;
    multi_source_dijkstra();
    std::vector<Node *> path = get_path_nodes();
    float cost = get_path_cost(path);
    std::cout << "Path found with cost: " << cost << std::endl;
    //std::cout << "num_goals_to_visit " << num_goals_to_visit << std::endl;
}

void Graph::get_path(int *vec, int n)
{
    if (path_found)
    {
        std::vector<Node *> path;
        path = get_path_nodes();
        if (n != path.size())
        {
            std::cout << "Wrong path nodes length" << std::endl;
        }
        else
        {
            for (int i = 0; i < n; ++i)
            {
                vec[i] = (*path[i]).sample_index;
            }
        }
    }
    else
    {
        std::cout << "No path found" << std::endl;
        for (std::size_t i = 0; i < n; ++i)
        {
            vec[i] = -1;
        }
    }
}

float Graph::get_path_cost()
{
    return shortest_path_cost;
}

// ===========================================================
// print functions for debugging
// ===========================================================
// private
void Graph::print_node(Node n)
{
    using namespace std;
    cout << "(" << n.path_index << ", ";
    cout << n.sample_index << ")";
    cout << " dist: " << n.dist;
    cout << " parent: ";
    cout << "(" << (*n.parent).path_index << ", ";
    cout << (*n.parent).sample_index << ")\n";
}

void Graph::print_nodes(std::vector<Node *> nodes)
{
    using namespace std;
    for (Node *node : nodes)
    {
        print_node(*node);
    }
    cout << endl;
}

void Graph::print_nodes(std::vector<Node> nodes)
{
    using namespace std;
    for (Node &node : nodes)
    {
        print_node(node);
    }
    cout << endl;
}

//public
void Graph::print_graph_data()
{
    using namespace std;
    for (auto &col : gd)
    {
        cout << "----------------" << endl;
        for (auto &val : col)
        {
            for (float &d : val)
            {
                cout << d << " ";
            }
            cout << endl;
        }
    }
}

void Graph::print_path()
{
    using namespace std;
    cout << "The most recent shortest path is:" << endl;
    print_nodes(get_path_nodes());
}

void Graph::print_graph()
{
    for (auto &p : na)
    {
        print_nodes(p);
    }
}

void Graph::set_graph_data(graph_data data)
{
    gd = data;
}
