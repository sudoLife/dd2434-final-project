import line_training
import networkx as nx
import numpy as np
import pandas as pd

def main():
    dataset = 'BlogCatalog'

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('datasets/' + dataset + '-dataset/data/edges.csv',
            #'BlogCatalog-dataset/data/edges.csv',
                     header=None, names=['src', 'dst'])

    # Create a graph from the DataFrame
    # 10312 nodes, 333983 edges
    G = nx.from_pandas_edgelist(
        df, source='src', target='dst', create_using=nx.Graph)

    final_embeddings = line_training.train(G, 128, isWeighted=False)    
    
    with open('Line/embedding_vectors/line_blog.npy', 'wb') as f:
        np.save(f, np.array(final_embeddings))
    #with open('line_blog.npy', 'rb') as f:
        #a = np.load(f)
        #b = np.load(f)
    #print(a, b)

if __name__ == '__main__':
    main()