#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <vector>
#include <limits> // infinity()

// to initialize distances for dijkstra
const float INF = std::numeric_limits<float>::infinity();

/**
 *  Container for a joint position.
 * Fixed length container more efficient but less flexible,
 * the length is the number of dofs of the robot, which can vary.
 */
typedef std::vector<float> joint_value;

/**
 * Container for graph data (joint positions).
 * The graph_data is the only place where the joint values are stored
 * (exept in python). The graph nodes contain pointers to this data.
 */
typedef std::vector<std::vector<joint_value>> graph_data;

/**
 *  A graph Node structure.
 * Holds a reference to the parant node, a reference to the associated data
 * and other info variables.
 */
struct Node
{
    unsigned int path_index;
    unsigned int sample_index;
    joint_value *jv;
    float dist;
    Node *parent;
    bool visited;
};

struct dijkstraSortFunction
{
    bool operator()(Node *n1, Node *n2) const
    {
        return (*n2).dist < (*n1).dist;
    }
};

/**
 * Container for all the nodes in the graph.
 * The node_array is the only variable containing the nodes.
 * All the other containers with nodes contain pointers to the nodes in graph.
 */
typedef std::vector<std::vector<Node>> node_array;

/** Sort function for shortest path algorithms.
 */
bool sort_function(Node *n1, Node *n2);

class Graph
{
    graph_data gd;
    node_array na;
    int MAX_ITER = 10000;
    bool path_found = false;
    std::vector<Node *> shortest_path;
    float shortest_path_cost = INF;
    int num_goals_to_visit = 0;
    unsigned int max_path_index = 0;
    std::vector<float> weights;

    void graph_data_to_node_array();
    void reset_node_array();
    void init_unvisited(std::vector<Node *> &uv);
    float cost_function(Node n1, Node n2);
    void visit(Node *node);
    std::vector<Node *> get_neighbors(Node *node);
    std::vector<Node *> get_path_nodes();
    float get_path_cost(std::vector<Node *> &path);
    void multi_source_bfs();
    void multi_source_dijkstra();

    void print_node(Node n);
    void print_nodes(std::vector<Node *> nodes);
    void print_nodes(std::vector<Node> nodes);

  public:
    void add_data_column(float *mat, int nrows, int ncols);
    void set_weights(float *vec_in, int n);
    void print_graph_data();
    void init();
    void reset();
    void run_dijkstra();
    void run_bfs();
    void get_path(int *vec, int n);
    float get_path_cost();
    void print_path();
    void print_graph();
    void set_graph_data(graph_data data);
};

#endif
