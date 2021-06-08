import numpy as np


class Stopper:
    def __init__(self, max_iter: int = 10, **kwargs):
        self.max_iter = max_iter
        self.n_iter = 0
        self.kwargs = kwargs
        self.last_loss = np.inf

    def new_training(self):
        self.n_iter = 0

    def stop(self, classifier, **kwargs) -> bool:
        print(f'Iteration: {self.n_iter}')
        kwargs.update(self.kwargs)
        self.n_iter += 1
        loss = classifier.L2_SVM_loss()[0][0]
        print(loss)
        if self.n_iter >= self.max_iter:
            return True
        if (self.last_loss - loss) / loss < 1e-3:
            return True
        self.last_loss = loss
        return False
