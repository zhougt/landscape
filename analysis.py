# coding: utf-8

import os
import itertools
import numpy as np
import pandas as pd
import networkx as nx

measures = [
    (nx.degree_centrality, 1),
    (nx.closeness_centrality, -1),
    (nx.betweenness_centrality, 1),
]


def parse_graph(graph_file, distance_threshold=200):
    dtypes = {
        'POINT_X': np.float64,
        'POINT_X': np.float64,
        'POINT_Z': np.float64}
    stops = pd.read_excel(graph_file, dtype=dtypes)

    graph = nx.Graph()
    for i in range(stops.shape[0]):
        node_attrs = {
            'name': stops['Name_GJCZ'][i],
            'routes': stops['XL_GJCZ'][i],
            'x': float(stops['POINT_X'][i]),
            'y': float(stops['POINT_Y'][i]),
            'z': float(stops['POINT_Z'][i])
        }
        graph.add_node(i, **node_attrs)

    for pair in itertools.combinations(graph.nodes, 2):
        n0 = graph.nodes[pair[0]]
        n1 = graph.nodes[pair[1]]

        dist = np.sqrt((n0['x'] - n1['x']) ** 2 + (n0['y'] - n1['y']) ** 2)

        routes0 = n0['routes'].split(';')
        routes1 = n1['routes'].split(';')
        routes = list(set(routes0) & set(routes1))

        edge_attrs = {
            'distance': float(dist),
            'routes': ';'.join(routes)
        }

        if dist <= distance_threshold or len(routes) > 0:
            print(u'Edge: {} and {}, dist: {}, routes: {}'.format(
                n0['name'], n1['name'], dist, ';'.join(routes)))
            graph.add_edge(pair[0], pair[1], **edge_attrs)

    save_file = '{}.gexf'.format(os.path.splitext(graph_file)[0])
    nx.write_gexf(graph, save_file)

    return graph


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
    # graph = nx.sedgewick_maze_graph()
    # graph = nx.tutte_graph()
    graph = parse_graph('stops_56.xlsx')

    best_combs, max_diff = find_comb(graph)
    print(best_combs)
