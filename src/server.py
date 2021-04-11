import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gmm import GaussianMixture
from utils import plot_PCA

class Server():
    def __init__(self, args, init_dataset, clients, output_dir):
        self.random_state = None
        if args.seed: self.random_state = (int(args.seed))
        self.model =  GaussianMixture(
            X=init_dataset,
            n_components=args.components,
            random_state=self.random_state,
            is_quiet=True,
            init_params='random'
        )

        self.init_dataset = init_dataset
        self.args = args
        self.rounds = args.rounds
        self.clients = clients
        self.fraction_clients = float(args.C)
        self.n_clients = int(args.K)
        self.n_clients_round = int(self.fraction_clients * self.n_clients)
        self.selected_clients = {}
        self.output_dir = output_dir

    def _select_round_clients(self, round):
        idxs_round_clients = np.random.choice(range(self.n_clients), self.n_clients_round, replace=False)
        selected_clients = []
        for idx in idxs_round_clients:
            selected_clients.append(self.clients[idx])

        self.selected_clients[round] = selected_clients
        return selected_clients

    def _set_parameters_from_clients_models(self, round_history):
        self.clients_means = []
        self.clients_covariances = []
        self.clients_weights = []
        
        for idx in round_history:
            history = round_history[idx]
            self.clients_means.append(history['means'])
            self.clients_covariances.append(history['covariances'])
            self.clients_weights.append(history['weights'])

        self.clients_means = np.array(self.clients_means)
        self.clients_covariances = np.array(self.clients_covariances)
        self.clients_weights = np.array(self.clients_weights)
    
    def start_round(self, round):
        selected_clients = self._select_round_clients(round)

        round_history = {}

        pbar = tqdm(selected_clients)
        for client in pbar:
            pbar.set_description('Round: {}/{} | Client: {}'.format(round+1, self.rounds, client.id))
            round_history[client.id] = client.fit(self.model)

        self._set_parameters_from_clients_models(round_history)

    def average_clients_models(self):
        gamma = 1 / self.n_clients_round # weight for each client (the same)
        
        [self.avg_clients_means] = np.sum(self.clients_means * gamma, axis=0)
        [self.avg_clients_covariances] = np.sum(self.clients_covariances * pow(gamma, 2), axis=0)
        [self.avg_clients_weights] = np.sum(self.clients_weights * gamma, axis=0)
        
        self.avg_clients_precisions_cholesky = self.model.compute_precision_cholesky(self.avg_clients_covariances, self.model.covariance_type)
        
        params = (self.avg_clients_weights, self.avg_clients_means, self.avg_clients_covariances, self.avg_clients_precisions_cholesky)
        self.model.set_parameters(params)

        self.avg_clients_precisions = self.model.precisions_

    def update_server_model(self):
        # The model must be regenerated with the new average parameters. It cannot simply be updated (it might be initialized again with wrong parameters)
        self.model =  GaussianMixture(
            X=self.init_dataset,
            n_components=self.args.components,
            random_state=self.random_state,
            is_quiet=True,
            weights_init=self.avg_clients_weights,
            means_init=self.avg_clients_means,
            precisions_init=self.avg_clients_precisions,
            init_params='random'
        )

    def plot(self, X, labels, round=None):
        path = './model'
        dir_name = os.path.join(self.output_dir, path)
        os.makedirs(dir_name, exist_ok=True)

        if round is None: filename = 'init'
        else: filename = 'round_{}'.format(round+1)
        dir_name = os.path.join(dir_name, filename)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        if X.shape[1] <= 2: self.args.plots_3d = 0
        if bool(self.args.plots_3d) == True:
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            pca_components = 3
        else:
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            pca_components = 2

        plot_PCA(ax1, X, labels, pca_components, self.args.soft, 'Dataset Clusters', random_state=self.random_state)
        plot_PCA(ax2, X, self.predict(X), pca_components, self.args.soft, 'Predicted Clusters', random_state=self.random_state)
        fig.savefig(dir_name, dpi=150)
        plt.close(fig)


    def predict(self, X):
        predicted_labels = self.model.predict_proba(X).tolist()
        predicted_labels = np.array(predicted_labels)

        return predicted_labels

