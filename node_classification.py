import numpy as np
import utils
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from utils import load_data_into_graph
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# NOTE: Set the dataset name and file with the embedding vectors
dataset = 'Flickr'  
embedding_vectors = 'DeepWalk-Flickr-1-epochs.npy'  

def main():
    G, node_count, label_count, node_labels = load_data_into_graph(dataset)

     # Load the embedding vectors
    input_model_file = embedding_vectors
    X = np.load(input_model_file)
    y = node_labels  # target labels (possibly in a multi-label format)

    utils.node_classification(
        G, X, y, multiple_labels=True, dataset=dataset, embedding_vectors_file=embedding_vectors)
    
if __name__ == "__main__":
    main()
