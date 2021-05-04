import os
import numpy as np
from tqdm import tqdm

from gmm import GaussianMixture

class Server():
    def __init__(self, args, init_dataset, clients, output_dir):
        self.random_state = None
        if args.seed: self.random_state = (int(args.seed))
        self.model =  GaussianMixture(
            X=init_dataset,
            n_components=args.components,
            random_state=self.random_state,
            is_quiet=True,
            init_params=args.init
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
        self.metrics_history = {
            'aic': [],
            'bic': [],
            'll': []
        }

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

        for client_id in round_history:
            parameters = round_history[client_id]['parameters']

            self.clients_means.append(parameters['means'][-1])
            self.clients_covariances.append(parameters['covariances'][-1])
            self.clients_weights.append(parameters['weights'][-1])

        self.clients_means = np.array(self.clients_means)
        self.clients_covariances = np.array(self.clients_covariances)
        self.clients_weights = np.array(self.clients_weights)

        return
    
    def _set_metrics_from_clients_models(self, round_history):
        self.clients_aic = []
        self.clients_bic = []
        self.clients_ll = []

        for client_id in round_history:
            metrics = round_history[client_id]['metrics']

            self.clients_aic.append(metrics['aic'][-1])
            self.clients_bic.append(metrics['bic'][-1])
            self.clients_ll.append(metrics['ll'][-1])

        self.clients_aic = np.array(self.clients_aic)
        self.clients_bic = np.array(self.clients_bic)
        self.clients_ll = np.array(self.clients_ll)

        return

    def start_round(self, round):
        selected_clients = self._select_round_clients(round)

        round_history = {}

        pbar = tqdm(selected_clients)
        for client in pbar:
            pbar.set_description('Round: {}/{} | Client: {}'.format(round+1, self.rounds, client.id))
            round_history[client.id] = client.fit(self.model, self.args.local_epochs)

            if pbar.iterable[-1] == client:
                pbar.set_description('Round: {}/{} | Completed'.format(round+1, self.rounds))
        
        self._set_parameters_from_clients_models(round_history)
        self._set_metrics_from_clients_models(round_history)

        return

    def average_clients_models(self):
        gamma = 1 / self.n_clients_round # weight for each client (the same)
        
        self.avg_clients_means = np.sum(self.clients_means * pow(gamma, 1), axis=0)
        self.avg_clients_covariances = np.sum(self.clients_covariances * pow(gamma, 2), axis=0)
        self.avg_clients_weights = np.sum(self.clients_weights * pow(gamma, 1), axis=0)
        
        self.avg_clients_precisions_cholesky = self.model.compute_precision_cholesky(self.avg_clients_covariances, self.model.covariance_type)
        
        params = (self.avg_clients_weights, self.avg_clients_means, self.avg_clients_covariances, self.avg_clients_precisions_cholesky)
        self.model.set_parameters(params)

        self.avg_clients_precisions = self.model.precisions_

        return

    def update_server_model(self):
        # The model must be regenerated with the new average parameters. It cannot simply be updated (it might be initialized again with wrong parameters)
        self.model =  GaussianMixture(
            X=self.init_dataset,
            n_components=self.args.components,
            random_state=self.random_state,
            is_quiet=True,
            init_params=self.args.init,
            weights_init=self.avg_clients_weights,
            means_init=self.avg_clients_means,
            precisions_init=self.avg_clients_precisions
        )

        return

    def average_clients_metrics(self):
        self.metrics_history['aic'].append(np.mean(self.clients_aic))
        self.metrics_history['bic'].append(np.mean(self.clients_bic))
        self.metrics_history['ll'].append(np.mean(self.clients_ll))

        return

    def plot(self, X, labels, round=None):
        self.model.plot(X, labels, self.args, self.output_dir, 'round', round)

        return

    def compute_init_metrics(self, X):
        self.metrics_history['aic'].append(self.model.aic(X))
        self.metrics_history['bic'].append(self.model.bic(X))
        self.metrics_history['ll'].append(self.model.score(X))

        return