import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import matplotlib as mpl
from numpy.core.fromnumeric import argmax
import numpy as np

import matplotlib.pyplot as plt

from utils import get_dataset, plot_PCA_2, show_img_clusters
from args_parser import parse_args

from gmm import GaussianMixture

if __name__ == '__main__':    
    args = parse_args(is_federated=False)
    if args.seed: random.seed(int(args.seed))

    train_dataset, train_dataset_labels, _ = get_dataset(args)

    print('\n')
    seed = None
    if args.seed: seed = (int(args.seed))
    global_model = GaussianMixture(
        X=train_dataset,
        n_components=args.components,
        random_state=seed
    )

    print('Configuration:')
    print('\n')

    global_model.fit(train_dataset, epochs=args.epochs)

    predicted_labels = global_model.predict_proba(train_dataset).tolist()

    flat_labels = [item for sublist in predicted_labels for item in sublist]
    if np.isnan(flat_labels).any():
        print('Error in predicting probabilities (nan): using hard bounds.')
        predicted_labels = global_model.predict(train_dataset).tolist()

    predicted_labels = np.array(predicted_labels)

    print('\nSaving images...')

    dir = os.path.dirname(__file__)
    path = '../save/img'
    filename = 'temp.png'

    dir_name = os.path.join(dir, path, filename)
    plt.figure()
    mydata = global_model.history_['ll']
    plt.plot(range(1, len(mydata)+1), mydata)
    plt.xlabel('Epochs')
    plt.ylabel('Log-Likelihood')
    plt.savefig(dir_name)

    filename = 'results.png'

    dir_name = os.path.join(dir, path, filename)
    fig, axs = plt.subplots(1, 2)
    plot_PCA_2(axs[0], train_dataset, train_dataset_labels, 'Dataset Clusters', random_state=args.seed)
    plot_PCA_2(axs[1], train_dataset, predicted_labels, 'Predicted Clusters', random_state=args.seed)
    fig.tight_layout()
    fig.savefig(dir_name, dpi=300)

    if args.dataset == 'custom': show_img_clusters(args, predicted_labels)
    print('Done.')

    

