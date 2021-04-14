import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np

from utils import get_dataset, plot_metric, prepare_output_dir
from utils import print_configuration, save_configuration
from args_parser import parse_args

from gmm import GaussianMixture

if __name__ == '__main__':    
    args = parse_args(is_federated=False)
    if args.seed: 
        random.seed(int(args.seed))
        np.random.RandomState(int(args.seed))

    output_dir = prepare_output_dir()

    train_dataset, train_dataset_labels, _ = get_dataset(args)

    print_configuration(args, train_dataset, False)
    save_configuration(args, train_dataset, output_dir, False)

    # Init the Gaussian Mixture Model
    seed = None
    if args.seed: seed = (int(args.seed))
    model = GaussianMixture(
        X=train_dataset,
        n_components=args.components,
        random_state=seed,
        init_params=args.init
    )

    model.fit(train_dataset, args.epochs, train_dataset_labels, args, output_dir)

    predicted_labels = model.predict_proba(train_dataset).tolist()
    predicted_labels = np.array(predicted_labels)

    print('\nSaving images...')

    metrics = model.history_['metrics']
    
    plot_metric(metrics['ll'], args.epochs, output_dir, 'Epochs', 'Log-Likelihood')
    plot_metric(metrics['aic'], args.epochs, output_dir, 'Epochs', 'AIC')
    plot_metric(metrics['bic'], args.epochs, output_dir, 'Epochs', 'BIC')

    print('Done.')

    

