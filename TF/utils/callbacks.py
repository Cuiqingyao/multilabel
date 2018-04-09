"""
Multi-label classification specific callbacks.
"""
import numpy as np

from keras.callbacks import Callback


class HammingLoss(Callback):
    def __init__(self, datasets, batch_size=128):
        self.datasets = datasets
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.params['metrics'] += ['hl', 'val_hl']

    def on_epoch_begin(self, epoch, logs={}, y_name=['Y0', 'Y1']):
        self.metrics = {}
        for dname, data in self.datasets.items():
            preds = self.model.predict(data, batch_size=self.batch_size)
            hl = np.hstack(
                [np.round(v) != data[y_name[k]]
                 for k, v in enumerate(preds)]).sum(axis=1).mean()
            self.metrics[dname] = hl

    def on_batch_begin(self, batch, logs={}):
        if 'train' in self.metrics:
            logs['hl'] = self.metrics['train']
        if 'valid' in self.metrics:
            logs['val_hl'] = self.metrics['valid']

    def on_epoch_end(self, epoch, logs={}):
        if 'train' in self.metrics:
            logs['hl'] = self.metrics['train']
        if 'valid' in self.metrics:
            logs['val_hl'] = self.metrics['valid']
