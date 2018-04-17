import itertools
import numpy as np
import networkx as nx

measures = [
    (nx.degree_centrality, 1),
    (nx.closeness_centrality, -1),
    (nx.betweenness_centrality, 1),
]


def score_nodes(graph, watch_nodes):
    scores = np.zeros((len(watch_nodes), len(measures)))

    for meas_idx, (meas_func, meas_weight) in enumerate(measures):
        meas_res = meas_func(graph)
        for node_idx, node in enumerate(watch_nodes):
            scores[node_idx, meas_idx] = meas_res[node] * meas_weight

    return scores


def find_comb(graph, watch_nodes=None, n_combs=2):
    if watch_nodes == None:
        watch_nodes = range(graph.number_of_nodes())

    new_node = 'new_node'
    best_combs = None
    max_diff = -np.inf

    scores = score_nodes(graph, watch_nodes)
    for comb in itertools.combinations(graph.nodes, n_combs):
        comb_nodes = list(comb)
        graph.add_node(new_node)
        for node in comb_nodes:
            graph.add_edge(new_node, node)

        comb_scores = score_nodes(graph, watch_nodes)
        diff = np.sum(comb_scores - scores)
        if diff > max_diff:
            max_diff = diff
            best_combs = comb_nodes

        graph.remove_node(new_node)
        print('Node combination: {}, diff: {}'.format(comb_nodes, diff))

    return best_combs, max_diff


if __name__ == '__main__':
    graph = nx.sedgewick_maze_graph()
    # graph = nx.tutte_graph()

    watch_nodes = [0, 2, 3]
    n_combs = 2
    find_comb(graph, watch_nodes, n_combs)
