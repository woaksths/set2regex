
import numpy as np
import torch
from seq2seq.util.checkpoint import Checkpoint

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, optimizer, epoch, step, input_vocab, output_vocab, expt_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            Checkpoint(model=model, optimizer=optimizer, epoch=epoch, step=step,
                       input_vocab=input_vocab, output_vocab=output_vocab).save(expt_dir +'/best_model')
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
            Checkpoint(model=model, optimizer=optimizer, epoch=epoch, step=step,
                       input_vocab=input_vocab, output_vocab=output_vocab).save(expt_dir +'/lowest_loss')
            self.counter = 0

