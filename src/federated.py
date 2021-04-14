import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np

from client import Client
from server import Server
from utils import get_dataset, prepare_output_dir
from utils import print_configuration, save_configuration
from args_parser import parse_args

if __name__ == '__main__':    
    args = parse_args(is_federated=True)
    if args.seed: random.seed(int(args.seed))
    
    output_dir = prepare_output_dir()   

    train_dataset, train_dataset_labels, clients_groups = get_dataset(args)

    print_configuration(args, train_dataset, True)
    save_configuration(args, train_dataset, output_dir, True)

    # Prepare clients
    clients = {}
    for idx_client in range(args.K):
        clients[idx_client] = Client(idx_client, train_dataset, clients_groups[idx_client])

    # Prepare server
    init_dataset_size = int(train_dataset.shape[0] * 0.005)
    init_dataset = train_dataset[np.random.choice(train_dataset.shape[0], init_dataset_size, replace=False)]
    server = Server(args, init_dataset, clients, output_dir)
    
    server.plot(train_dataset, train_dataset_labels)
    for round in range(args.rounds):
        server.start_round(round)
        server.average_clients_models()
        server.update_server_model()
        if (round+1) % args.plots_step == 0: server.plot(train_dataset, train_dataset_labels, round)

    predicted_labels = server.model.predict(train_dataset)

    print('\nSaving images...')
    
    # # Final plot
    # filename = 'results.png'
    # dir_name = os.path.join(output_dir, filename)
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # if train_dataset.shape[1] <= 2: args.plots_3d = 0
    # if bool(args.plots_3d) == True:
    #     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    #     ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #     pca_components = 3
    # else:
    #     ax1 = fig.add_subplot(1, 2, 1)
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     pca_components = 2

    # plot_PCA(ax1, train_dataset, train_dataset_labels, pca_components, args.soft, 'Dataset Clusters', random_state=args.seed)
    # plot_PCA(ax2, train_dataset, predicted_labels, pca_components, args.soft, 'Predicted Clusters', random_state=args.seed)
    # fig.savefig(dir_name, dpi=300)
    # plt.close(fig)

    print('Done.')
