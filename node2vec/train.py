import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import os
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier

# Define global constant variables
directed = False
p = 2.0
q = 1.0
num_walks = 10
walk_length = 80
emb_size = 512
iteration = 1

def load_graph(filename, id_list):
	'''
	Load adjacent matrix from the file
	'''
	if directed:
		g = nx.DiGraph();
	else:
		g = nx.Graph();

	with open(filename, 'r') as f:
		line = f.readline();
		while line:
			line_split = line.split();
			
			if line_split[0] in id_list and line_split[1] in id_list and line_split[0]!=line_split[1]: 
				g.add_edge(line_split[0], line_split[1])
				g[line_split[0]][line_split[1]]['weight'] = 1

			line = f.readline();

	return g;

def load_features(filename):
	'''
	Load node name and its label from file
	'''
	ids, labels = [], []
	with open(filename, 'r') as f:
		line = f.readline();
		while line:
			line_split = line.split();

			ids.append(line_split[0]);
			labels.append(line_split[-1]);
			line = f.readline();

	return ids, labels



def main():
	edge_path = 'data_cora/cora.content'
	label_path = 'data_cora/cora.cites'
	model_path = './output.model'

	# load feature and adjacent matrix from file
	id_list, labels = load_features(edge_path)
	g = load_graph(label_path, id_list)

	# Add isolated nodes to the graph
	for node in id_list:
		if not g.has_node(node):
			g.add_node(node);

	# Main body for node2vec
	if os.path.isfile(model_path):
		model = Word2Vec.load(model_path);
		print ('load model successfully')
	else: 
		alias_nodes, alias_edges = preprocess_transition_probs(g, directed)
		
		walks = [];
		idx_total = []
		for i in range(num_walks):
			r = range(len(id_list))
			np.random.shuffle(r)
			idx_total+=r
			for node in [id_list[j] for j in r]:
				walks.append(node2vec_walk(g, node, alias_nodes, alias_edges, walk_length))

		model = Word2Vec(walks, size=emb_size, min_count=0, sg=0, iter=iteration)
		model.save('output.model')

	# transform word2vec model to dict format
	w2v = dict(zip(model.wv.index2word, model.wv.syn0))

	key_list = w2v.keys();
	x_list = np.array(w2v.values())
	y_list = [labels[id_list.index(key)] for key in key_list]

	lab_enc = LabelEncoder()
	hot_enc = OneHotEncoder();
	y_list = lab_enc.fit_transform(y_list)
	y_list_res = [[c] for c in y_list]
	y_list_onehot = hot_enc.fit_transform(y_list_res).toarray()
	
	X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size = 0.2)
	neigh = ExtraTreesClassifier()
	neigh.fit(X_train, y_train)
	preds = neigh.predict(X_test)
	print (preds)
	acc = sum(np.equal(preds, y_test))

	print ('accuray is '+ str(float(acc)/len(preds)))
	
	
if __name__=='__main__':
	main()