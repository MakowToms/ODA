
class Stopper:
    def __init__(self, max_iter: int = 10, **kwargs):
        self.max_iter = max_iter
        self.n_iter = 0
        self.kwargs = kwargs

    def new_training(self):
        self.n_iter = 0

    def stop(self, **kwargs) -> bool:
        kwargs.update(self.kwargs)
        self.n_iter += 1
        if self.n_iter >= self.max_iter:
            return True
        return False

