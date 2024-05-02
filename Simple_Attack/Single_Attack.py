import torch as th
import torch.nn.functional as F
from torch.nn.modules.module import Module

class SINGLE_ATTACK(Module):
    def __init__(self, model):
        super(SINGLE_ATTACK, self).__init__()
        self.model = model

    def attack(self, Batch_data, beta, epsilon=0.5):#0.005

        Batch_data.x.requires_grad = True
        Batch_data.x0.requires_grad = True
        out_labels, cl_loss, y = self.model(Batch_data)
        # finalloss = F.nll_loss(out_labels, y)
        finalloss = F.nll_loss(out_labels, y) + beta * cl_loss
        loss = finalloss
        loss.backward()

        # grad_x = th.autograd.grad(outputs=loss, inputs=Batch_data.x, retain_graph=True, allow_unused=True)
        # grad_x0 = th.autograd.grad(outputs=loss, inputs=Batch_data.x0, allow_unused=True)

        grad_x = Batch_data.x.grad
        grad_x0 = Batch_data.x0.grad

        norm_x = th.norm(grad_x)
        norm_x0 = th.norm(grad_x)

        #####FGSM
        # Batch_data.x.data += grad_x.data.sign() * epsilon
        # Batch_data.x0.data += grad_x0.data.sign() * epsilon


        Batch_data.x.data += epsilon * grad_x / norm_x
        Batch_data.x0.data += epsilon * grad_x0 / norm_x0