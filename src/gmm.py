import sklearn.mixture
import numpy as np
import torch
from tqdm import tqdm
from scipy import linalg

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


class GaussianMixture(sklearn.mixture.GaussianMixture):

    def __init__(
        self,
        X,
        n_components=3,
        covariance_type='full',
        weights_init=None,
        means_init=None,
        precisions_init=None,
        covariances_init=None,
        init_params='kmeans',
        tol=1e-3,
        random_state=None,
        is_quiet=False
    ):

        if torch.is_tensor(X):
            X = np.array(X.tolist())

        do_init = (weights_init is None) and (means_init is None) and (precisions_init is None) and (covariances_init is None)
        if do_init:
            _init_model = sklearn.mixture.GaussianMixture(
                n_components=n_components,
                tol=tol,
                covariance_type=covariance_type,
                random_state=random_state,
                init_params=init_params
            )

            # Responsibilities are found through KMeans or randomly assigned, from responsibilities the gaussian parameters are estimated (precisions_ is not calculated)
            _init_model._initialize_parameters(X, np.random.RandomState(random_state))
            # The gaussian parameters are fed into _set_parameters() which computes also precisions_ (the others remain the same)
            _init_model._set_parameters(_init_model._get_parameters())
            
            weights_init=_init_model.weights_
            means_init=_init_model.means_
            precisions_init=_init_model.precisions_
            covariances_init=_init_model.covariances_

        super().__init__(
            n_components=n_components,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            tol=tol,
            init_params=init_params,
            covariance_type=covariance_type,
            random_state=random_state,
            warm_start=True,
            max_iter=1
        )
        self._is_quiet = is_quiet
        # The gaussian parameters are recomputed by KMeans or randomly, but since the init parameters are given they are discarded (covariances_ is not generated)
        self._initialize_parameters(X, np.random.RandomState(random_state))
        # covariances_ is copied from the initial model (since it has it)
        self.covariances_ = covariances_init
        # precisions_ is computed as before
        self._set_parameters(self._get_parameters())

    def fit(self, X, epochs=1):

        if torch.is_tensor(X):
            X = np.array(X.tolist())

        self.history_ = {
            'epochs': epochs,
            'aic': [],
            'bic': [],
            'll': [],
            'converged': [],
            'means': [],
            'covariances': [],
            'weights': []
        }

        #is_not_converged = (not hasattr(self, 'converged_')) or (self.converged_ is False)

        pbar = tqdm(range(epochs), disable=self._is_quiet)
        for epoch in pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch+1, epochs))
            super().fit(X)
            self.history_['aic'].append(self.aic(X))
            self.history_['bic'].append(self.bic(X))
            self.history_['ll'].append(self.lower_bound_)
            self.history_['converged'].append(self.converged_)
            self.history_['means'].append(self.means_)
            self.history_['weights'].append(self.weights_)
            self.history_['covariances'].append(self.covariances_)

        if not self._is_quiet:
            if self.converged_:
                print('\nThe model successfully converged.')
            else:
                print('\nThe model did NOT converge.')

        return self.history_

    def get_parameters(self):
        parameters = self._get_parameters()
        return parameters

    def set_parameters(self, params):
        self._set_parameters(params)

    def compute_precision_cholesky(self, covariances, covariance_type):
        """Compute the Cholesky decomposition of the precisions.
        Parameters
        ----------
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.
        Returns
        -------
        precisions_cholesky : array-like
            The cholesky decomposition of sample precisions of the current
            components. The shape depends of the covariance_type.
        """
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar.")

        if covariance_type == 'full':
            n_components, n_features, _ = covariances.shape
            precisions_chol = np.empty((n_components, n_features, n_features))
            for k, covariance in enumerate(covariances):
                try:
                    cov_chol = linalg.cholesky(covariance, lower=True)
                except linalg.LinAlgError:
                    raise ValueError(estimate_precision_error_message)
                precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                            np.eye(n_features),
                                                            lower=True).T
        elif covariance_type == 'tied':
            _, n_features = covariances.shape
            try:
                cov_chol = linalg.cholesky(covariances, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                    lower=True).T
        else:
            if np.any(np.less_equal(covariances, 0.0)):
                raise ValueError(estimate_precision_error_message)
            precisions_chol = 1. / np.sqrt(covariances)
        return precisions_chol