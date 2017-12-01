import os
import time
import shutil
import numpy as np
import torch

from inflection import titleize

from collections import OrderedDict
from .meters import AverageMeter


class Callback(object):

    def __init__(self):
        pass

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_step_begin(self, size, mode=''):
        pass

    def on_batch_begin(self, batch, batch_size):
        pass

    def on_batch_end(self, metrics):
        pass

    def on_step_end(self):
        pass

    def on_epoch_end(self, metrics):
        pass

    def on_end(self):
        pass

    def set_params(self, net=None, optimizer=None, criterion=None):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion


class Compose(Callback):

    def __init__(self, callbacks=[]):
        if len(callbacks) and not all([isinstance(c, Callback)
                                       for c in callbacks]):
            raise ValueError('All callbacks must be an instance of Callback')

        self.callbacks = callbacks

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        for callback in self.callbacks:
            callback.on_begin(start_epoch, end_epoch, metrics_name)

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_step_begin(self, size, mode=''):
        for callback in self.callbacks:
            callback.on_step_begin(size, mode=mode)

    def on_step_end(self):
        for callback in self.callbacks:
            callback.on_step_end()

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(metrics)

    def on_batch_begin(self, batch, batch_size):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, batch_size)

    def on_batch_end(self, metrics):
        for callback in self.callbacks:
            callback.on_batch_end(metrics)

    def set_params(self, net=None, optimizer=None, criterion=None):
        for callback in self.callbacks:
            callback.set_params(net, optimizer, criterion)


class Progbar(Callback):

    def __init__(self, print_freq=0):
        self.print_freq = print_freq

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        self.end_epoch = end_epoch
        self.data_time = AverageMeter()

    def on_step_begin(self, size, mode=''):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.end = time.time()
        self.mode = mode
        self.size = size

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_batch_begin(self, batch, batch_size):
        self.batch = batch
        self.data_time.update(time.time() - self.end)

    def on_batch_end(self, metrics):
        # measure elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

        if self.batch % self.print_freq == 0:
            msg = []
            if self.mode.startswith('train'):
                msg += ['Epoch: [{0}][{1}/{2}]\t'.format(self.epoch,
                                                         self.batch,
                                                         self.size)]
            else:
                msg += ['{0}: [{1}/{2}]\t'.format(titleize(self.mode),
                                                  self.batch,
                                                  self.size)]

            msg += ['Time {0.val:.3f} ({0.avg:.3f})\t'.format(self.batch_time)]
            msg += ['Data {0.val:.3f} ({0.avg:.3f})\t'.format(self.data_time)]

            # Add metrics alongsise with the loss
            msg += ['{0} {1.val:.3f} ({1.avg:.3f})\t'
                    .format(titleize(name), meter)
                    for name, meter in metrics.items()]

            print(''.join(msg))


class ModelCheckpoint(Callback):

    def __init__(self, filepath, monitor='val_loss', mode='min', save_best=True,
                 history=None):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best = save_best
        self.history = history

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
            op = np.min
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
            op = np.max
        else:
            raise ValueError('mode not recognized.')

        if history:
            self.best = op(self.history[monitor])
        else:
            self.history = OrderedDict()

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        if not len(self.history):
            self.history['epochs'] = np.arange(end_epoch)
            for name in metrics_name:
                self.history[name] = []

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_epoch_end(self, metrics):
        # Keep track of values
        for name, meter in metrics.items():
            self.history[name].append(meter.avg)

        is_best = self.monitor_op(metrics[self.monitor].avg, self.best)
        self.best = metrics[self.monitor].avg if is_best else self.best

        state = {
            'epoch': self.epoch + 1,
            'net': self.net,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'criterion': self.criterion.state_dict(),
            'best_{}'.format(self.monitor): self.best
        }
        state['history'] = self.history

        filepath = self.filepath.format(epoch=self.epoch, **metrics)

        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath,
                            '{}-best{}'.format(*os.path.splitext(filepath)))


class LearningRateScheduler(Callback):

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_begin(self, epoch):
        self.scheduler.step()


class Visdom(Callback):
    def __init__(self, server='http://localhost', port=8097, env='main',
                 history=OrderedDict()):
        from visdom import Visdom
        self.viz = Visdom(server=server, port=port, env=env)
        self.history = history

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        self.modes, self.metrics = zip(*[metric.split('_', 1)
                                         for metric in metrics_name])

        self.metrics = list(OrderedDict.fromkeys(self.metrics))
        self.modes = list(OrderedDict.fromkeys(self.modes))
        self.modes = self.modes or ['']

        self.viz_windows = {m: None for m in self.metrics}

        self.opts = {m: dict(title=titleize(m), ylabel=titleize(m),
                             xlabel='Epoch',
                             legend=[titleize(mode)
                                     for mode in self.modes])
                     for m in self.metrics}

        if not len(self.history):
            for name in metrics_name:
                self.history[name] = np.zeros(end_epoch)
        self.history['epochs'] = np.arange(1, end_epoch + 1)

        if start_epoch != 0:
            for m in self.metrics:
                self.viz_windows[m] = self.viz.line(
                    X=np.column_stack(
                        [self.history['epochs'][0:start_epoch]
                         for _ in self.modes]),
                    Y=np.column_stack(
                        [self.history['{}_{}'.format(mode, m)][0:start_epoch]
                         for mode in self.modes]),
                    opts=self.opts[m])

    def on_epoch_begin(self, epoch):
        self.epoch = epoch

    def on_epoch_end(self, metrics):
        # Keep track of values
        for name, meter in metrics.items():
            self.history[name][self.epoch] = meter.avg

        for m in self.metrics:
            if self.viz_windows[m] is None:
                self.viz_windows[m] = self.viz.line(
                    X=np.column_stack(
                        [self.history['epochs'][0:self.epoch + 1]
                         for _ in self.modes]),
                    Y=np.column_stack(
                        [self.history['{}_{}'.format(mode, m)][0:self.epoch + 1]
                         for mode in self.modes]),
                    opts=self.opts[m])
            else:
                self.viz.line(
                    X=np.column_stack(
                        [self.history['epochs'][0:self.epoch + 1]
                         for _ in self.modes]),
                    Y=np.column_stack(
                        [self.history['{}_{}'.format(mode, m)][0:self.epoch + 1]
                         for mode in self.modes]),
                    win=self.viz_windows[m],
                    update='replace')
