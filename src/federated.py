import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np

from client import Client
from server import Server
from utils import get_dataset, plot_metric, prepare_output_dir
from utils import print_configuration, save_configuration
from args_parser import parse_args

if __name__ == '__main__':    
    args = parse_args(is_federated=True)
    if args.seed: 
        random.seed(int(args.seed))
        np.random.RandomState(int(args.seed))
    
    output_dir = prepare_output_dir()   

    train_dataset, train_dataset_labels, clients_groups = get_dataset(args)

    print_configuration(args, train_dataset, True)
    save_configuration(args, train_dataset, output_dir, True)

    # Prepare clients
    clients = {}
    for idx_client in range(args.K):
        clients[idx_client] = Client(idx_client, train_dataset, clients_groups[idx_client])

    # Prepare server --> init_dataset is given by 0.5% of the train_dataset randomly sampled
    init_dataset_size = int(train_dataset.shape[0] * 0.005)
    init_dataset = train_dataset[np.random.choice(train_dataset.shape[0], init_dataset_size, replace=False)]
    server = Server(args, init_dataset, clients, output_dir)
    
    server.compute_init_metrics(train_dataset)
    server.plot(train_dataset, train_dataset_labels)
    for round in range(args.rounds):
        server.start_round(round)
        server.average_clients_models(use_hellinger_distance=True, update_reference=False)
        server.average_clients_metrics()
        server.update_server_model()

        if (round+1) % args.plots_step == 0: server.plot(train_dataset, train_dataset_labels, round)

    predicted_labels = server.model.predict(train_dataset)

    print('\nSaving images...')
    
    metrics = server.metrics_history
    
    plot_metric(metrics['ll'], args.rounds, output_dir, 'Rounds', 'Log-Likelihood')
    plot_metric(metrics['aic'], args.rounds, output_dir, 'Rounds', 'AIC')
    plot_metric(metrics['bic'], args.rounds, output_dir, 'Rounds', 'BIC')

    print('Done.')
