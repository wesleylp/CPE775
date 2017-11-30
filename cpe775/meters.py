import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if np.isnan(val):
            print('nan found. Skipping')
            return

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class History(AverageMeter):
    """ Stores values and computes some metrics """
    def reset(self):
        super(History, self).reset()
        self.vals = []

    def update(self, val, n=1):
        super(History, self).update(val, n)
        self.vals.append(val * n)
