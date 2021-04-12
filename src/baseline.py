import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import matplotlib.pyplot as plt

from utils import get_dataset, plot_PCA, prepare_output_dir
from args_parser import parse_args

from gmm import GaussianMixture

if __name__ == '__main__':    
    args = parse_args(is_federated=False)
    if args.seed: random.seed(int(args.seed))

    output_dir = prepare_output_dir()

    train_dataset, train_dataset_labels, _ = get_dataset(args)

    print('\n')
    print('Configuration:')
    print('\n')

    # Init the Gaussian Mixture Model
    seed = None
    if args.seed: seed = (int(args.seed))
    model = GaussianMixture(
        X=train_dataset,
        n_components=args.components,
        random_state=seed,
        init_params=args.init
    )

    model.fit(train_dataset, epochs=args.epochs)

    predicted_labels = model.predict_proba(train_dataset).tolist()
    predicted_labels = np.array(predicted_labels)

    print('\nSaving images...')

    # filename = 'temp.png'
    # dir_name = os.path.join(dir, path, filename)
    # plt.figure()
    # mydata = model.history_['ll']
    # plt.plot(range(1, len(mydata)+1), mydata)
    # plt.xlabel('Epochs')
    # plt.ylabel('Log-Likelihood')
    # plt.savefig(dir_name)

    # Final plot
    filename = 'results.png'
    dir_name = os.path.join(output_dir, filename)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    if train_dataset.shape[1] <= 2: args.plots_3d = 0
    if bool(args.plots_3d) == True:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        pca_components = 3
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        pca_components = 2

    plot_PCA(ax1, train_dataset, train_dataset_labels, pca_components, args.soft, 'Dataset Clusters', random_state=args.seed)
    plot_PCA(ax2, train_dataset, predicted_labels, pca_components, args.soft, 'Predicted Clusters', random_state=args.seed)
    fig.savefig(dir_name, dpi=300)
    plt.close(fig)

    print('Done.')

    

