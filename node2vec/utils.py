import numpy as np
import random

def preprocess_transition_probs(g, directed = False, p=1, q=1):
	'''
	Preprocessing of transition probs for guiding random walks
	'''
	alias_nodes, alias_edges = {}, {};
	for node in g.nodes():
		probs = [g[node][nei]['weight'] for nei in sorted(g.neighbors(node))]
		norm_const = sum(probs)
		norm_probs = [float(prob)/norm_const for prob in probs]
		alias_nodes[node] = get_alias_nodes(norm_probs)

	if directed:
		for edge in g.edges():
			alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
	else:
		for edge in g.edges():
			alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
			alias_edges[(edge[1], edge[0])] = get_alias_edges(g, edge[1], edge[0], p, q)

	return alias_nodes, alias_edges

def get_alias_edges(g, src, dest, p=1, q=1):
	'''
	get the alias edge setup lists for a given edge
	'''
	probs = [];
	for nei in sorted(g.neighbors(dest)):
		if nei==src:
			probs.append(g[dest][nei]['weight']/p)
		elif g.has_edge(nei, src):
			probs.append(g[dest][nei]['weight'])
		else:
			probs.append(g[dest][nei]['weight']/q)
	norm_probs = [float(prob)/sum(probs) for prob in probs]
	return get_alias_nodes(probs)


def get_alias_nodes(probs):
	'''
	Compute utility lists for non-uniform samplling from discrete distribution
	'''
	l = len(probs)
	a, b = np.zeros(l), np.zeros(l, dtype=np.int)
	small, large = [], []

	for i, prob in enumerate(probs):
		a[i] = l*prob
		if a[i]<1.0:
			small.append(i)
		else:
			large.append(i)

	while small and large:
		sma, lar = small.pop(), large.pop()
		b[sma] = lar
		a[lar]+=a[sma]-1.0
		if a[lar]<1.0:
			small.append(lar)
		else:
			large.append(lar)

	return b, a

def node2vec_walk(g, start, alias_nodes, alias_edges, walk_length=30):
	'''
	Reaptly simulate random walk for each node
	'''
	path = [start]
	while len(path)<walk_length:
		node = path[-1]
		neis = sorted(g.neighbors(node))
		if len(neis)>0:
			if len(path)==1:
				l = len(alias_nodes[node][0])
				idx = int(np.floor(np.random.rand()*l))
				if np.random.rand()<alias_nodes[node][1][idx]:
					path.append(neis[idx])
				else:
					path.append(neis[alias_nodes[node][0][idx]])
			else:
				prev = path[-2]
				l = len(alias_edges[(prev, node)][0])
				idx = int(np.floor(np.random.rand()*l))
				if np.random.rand()<alias_edges[(prev, node)][1][idx]:
					path.append(neis[idx])
				else:
					path.append(neis[alias_edges[(prev, node)][0][idx]])
		else:
			break;

	return path 