import os
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from sampling import sample_iid, sample_non_iid
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import colorsys
import json

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    dir = os.path.dirname(__file__)

    if args.dataset == 'blob':
        seed = None
        if args.seed:
            seed = int(args.seed)

        data, labels = datasets.make_blobs(
            n_samples=args.samples,
            n_features=args.features,
            centers=args.components,
            random_state=seed,
            shuffle=False
        )

        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        train_dataset = torch.Tensor(data)

        lb = preprocessing.LabelBinarizer()
        lb.fit(labels)
        labels = lb.transform(labels)
        train_dataset_labels = labels
        
        clients_groups = None
        if hasattr(args, 'K'):
            if hasattr(args, 'S') and args.S is not None:
                clients_groups = sample_non_iid(train_dataset, args.K, shards_per_client=5)
            else: 
                clients_groups = sample_iid(train_dataset, args.K)

    elif args.dataset == 's1':
        seed = None
        if args.seed:
            seed = int(args.seed)

        path = '../data/s/s1.txt'
        data = pd.read_csv(os.path.join(dir, path), header=None, delimiter='\t')

        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        train_dataset = torch.Tensor(data)

        train_dataset_labels = np.ones((train_dataset.shape[0], 1))
        
        clients_groups = None
        if hasattr(args, 'K'):
            if hasattr(args, 'S') and args.S is not None:
                clients_groups = sample_non_iid(train_dataset, args.K, shards_per_client=5)
            else: 
                clients_groups = sample_iid(train_dataset, args.K)

    return train_dataset, train_dataset_labels, clients_groups

def print_configuration(args, dataset, is_federated):
    print('\nCONFIGURATION')
    print('--------------------------------------------------')

    print(f'Mode: FEDERATED') if is_federated else print(f'Mode: BASELINE')
    
    # Common
    print(f'Clusters: {args.components}')
    print(f'\tSoft Clustering: {bool(args.soft)}')
    print(f'Dataset: {args.dataset}')
    print(f'\tFeatures: {dataset.shape[1]}')
    print(f'\tInstances: {dataset.shape[0]}')
    partition_type = 'non-IID' if (hasattr(args, 'S') and args.S is not None) else 'IID'
    print(f'\tPartition: {partition_type}') if is_federated else print(f'\tPartition: NULL')
    print(f'Initialization: {args.init}')
    print(f'Random Seed: {args.seed}')
    print(f'Plots: 3D') if bool(args.plots_3d) else print(f'Plots: 2D')
    print(f'\tPlotting Step: {args.plots_step}')

    if is_federated:
        # Federated
        print(f'Rounds: {args.rounds}')
        print(f'\tLocal Epochs: 1')
        print(f'Clients: {args.K}')
        print(f'\tClients fraction: {args.C * 100}%')
        print(f'\tClients per round: {int(args.K * args.C)}')
        print(f'\tData instances per client: {int(dataset.shape[0] / args.K)}')
    else:
        # Baseline
        print(f'Epochs: {args.epochs}')

    print('--------------------------------------------------')
    print('\n')

    return

def save_configuration(args, dataset, output_dir, is_federated):
    output_dir = str(output_dir).split('\\')[-1]

    partition_type = 'non-IID' if (hasattr(args, 'S') and args.S is not None) else 'IID'

    configuration = {
        'execution_time': output_dir,
        'mode': 'FEDERATED' if is_federated else 'BASELINE',
        'clusters': args.components,
        'soft_clustering': bool(args.soft),
        'dataset': {
            'name': args.dataset,
            'features': dataset.shape[1],
            'instances': dataset.shape[0],
            'partition': partition_type if is_federated else 'null',
        },
        'initialization': args.init,
        'random_seed': args.seed,
        'plots': {
            'type': '3D' if args.plots_3d else '2D',
            'step_size': args.plots_step
        }
    }

    if is_federated:
        # Federated
        configuration['dataset']['data_instances_per_client'] = int(dataset.shape[0] / args.K)
        configuration['rounds'] = args.rounds
        configuration['local_epochs'] = 1
        configuration['clients'] = {
            'total': args.K,
            'round_fraction_perc': str(args.C * 100) + '%',
            'round_fraction_num': int(args.K * args.C), 
        }
    else:
        # Baseline
        configuration['epochs'] = args.epochs

    filename = 'config.json'
    path =   r'output/' + output_dir + '/' + filename
    with open(path, 'w') as f:
        json.dump(configuration, f)

    return

def plot_PCA(ax, X, labels, pca_components=2, soft_clustering=True, title=None, random_state=None):  
    if X.shape[1] > 1:  
        if pca_components > 2: pca_components = 3
        else: pca_components = 2

        if random_state:
            random_state = int(random_state)
        pca = PCA(n_components=pca_components, random_state=random_state)
        pca.fit(X)
        pca_data = pca.transform(X)

        pc1 = pca_data[:, 0]
        pc2 = pca_data[:, 1]
        if pca_components == 3: pc3 = pca_data[:, 2]

        if not bool(soft_clustering):
            idxs = np.argmax(labels, axis=1)
            labels = np.zeros(labels.shape)
            for i in range(labels.shape[0]):
                labels[i, idxs[i]] = 1

        N = labels.shape[1]  
        HSV_tuples = [(i*1.0/N, 0.75, 0.75) for i in range(N)]
        RGB_tuples = np.array(list(map(lambda i: colorsys.hsv_to_rgb(*i), HSV_tuples)))
        colors = np.matmul(labels, RGB_tuples)

        if pca_components == 2:
            ax.set_aspect(1)
            ax.scatter(pc1, pc2, s=0.1, c=colors)
        else:
            ax.scatter(pc1, pc2, pc3, s=0.1, c=colors)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if pca_components == 3: ax.set_zlabel('PC3')

        if title: ax.set_title(title)   
    else:
        print('Data have only 1 feature. PCA cannot be applied.')

    return

def plot_metric(metric, n_iterations, output_dir, xLabel, yLabel):
    filename = str(yLabel).lower().replace('-', '_') + '.png'
    dir_name = os.path.join(output_dir, filename)

    ax = plt.figure().gca()

    if n_iterations != len(metric):
        x = np.arange(start=0, stop=n_iterations+1)
    else:
        x = np.arange(start=1, stop=n_iterations+1)
    y = metric
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.plot(x, y)
    
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(yLabel + ' vs. ' + xLabel)
    plt.tight_layout()

    plt.savefig(dir_name, dpi=150)
    plt.close()

    return

def prepare_output_dir():
    dir = os.path.dirname(__file__)
    path = '../output'
    dir_name = os.path.join(dir, path)
    os.makedirs(dir_name, exist_ok=True)

    date = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    output_dir = os.path.join(dir, path, date)
    os.makedirs(output_dir) 

    return output_dir