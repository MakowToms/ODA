import numpy as np
from time import time


class Stopper:
    def __init__(self, max_iter: int = 10, **kwargs):
        self.max_iter = max_iter
        self.n_iter = 0
        self.kwargs = kwargs
        self.best5 = np.ones([5]) * np.inf
        self.loss_history = []
        self.start_time = time()
        self.time_history = []

    def new_training(self):
        self.n_iter = 0

    def stop(self, classifier, **kwargs) -> bool:
        print(f'Iteration: {self.n_iter}')
        kwargs.update(self.kwargs)
        self.n_iter += 1
        loss = classifier.L2_SVM_loss()[0][0]
        self.loss_history.append(loss)
        self.time_history.append(time() - self.start_time)
        print(loss)
        if self.n_iter >= self.max_iter:
            return True
        if (max(self.best5) - loss) / loss < 1e-3:
            return True
        if loss < self.best5.max():
            self.best5[self.best5.argmax()] = loss
        return False
