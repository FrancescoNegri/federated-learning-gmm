import copy
import numpy as np

class Client():
    def __init__(self, id, global_dataset, idxs):
        self.id = id
        self.dataset = []
        for idx in idxs:
            self.dataset.append(global_dataset[idx].tolist())
        self.dataset = np.array(self.dataset)

    def fit(self, global_model):
        local_model = copy.deepcopy(global_model)
        local_model.fit(self.dataset, epochs=1)

        return local_model.history_