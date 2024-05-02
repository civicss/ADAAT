# -*- coding = utf-8 -*-
# @Time: 2023/3/7 10:13
# @Author: YZQ
# @File: early2stopping.py
# @Software: PyCharm
import numpy as np
import torch

class Early2Stopping:

    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1=0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,acc1,acc2,pre1,pre2,rec1,rec2,F1,F2,model,modelname,str):

        score = (accs+F1+F2) /3

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.acc1 = acc1
            self.acc2 = acc2
            self.pre1 = pre1
            self.pre2 = pre2
            self.rec1 = rec1
            self.rec2 = rec2
            self.F1 = F1
            self.F2 = F2

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                '''
                print("BEST Accuracy: {:.3f}|NR F1: {:.3f}|FR F1: {:.3f}|TR F1: {:.3f}|UR F1: {:.3f}"
                      .format(self.accs,self.F1,self.F2,self.F3,self.F4))
                '''
                print("BEST SCORE:{:.4f}| Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                      "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(self.best_score, self.accs, self.acc1, self.acc2, self.pre1, self.pre2, self.rec1,
                              self.rec2, self.F1, self.F2))

        else:
            self.best_score = score
            self.accs = accs
            self.acc1 = acc1
            self.acc2 = acc2
            self.pre1 = pre1
            self.pre2 = pre2
            self.rec1 = rec1
            self.rec2 = rec2
            self.F1 = F1
            self.F2 = F2
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname, str):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(), modelname + str + '.m')
        self.val_loss_min = val_loss