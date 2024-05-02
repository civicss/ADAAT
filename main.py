# --------- GACL meathod ------------
# Time 2022 

import sys,os
from numpy.matrixlib.defmatrix import matrix
sys.path.append(os.getcwd())
from Process.process import * 
#from Process.process_user import *
import torch as th
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from others.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from others.evaluate import *
from torch_geometric.nn import GCNConv
import copy
import random
import time
from Simple_Attack.Single_Attack import SINGLE_ATTACK

time_begin = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print("time_begin!!!!!!!!!!!!!!!!", time_begin)

def setup_seed(seed):
     th.manual_seed(seed)
     th.cuda.manual_seed_all(seed) 
     np.random.seed(seed)
     random.seed(seed)
     th.backends.cudnn.deterministic = True

setup_seed(12022)
#2022,12022,7869,17869
print("SEED!!!!!!!!!!!!!!!!!!!!!!!!!!!!",12022)

class hard_fc(th.nn.Module):
    def __init__(self, d_in,d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)

    def forward(self, x):

        residual = x
        # print("w_1!!!!!!!!!!!", self.w_1.weight)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.2, emb_name='hard_fc1.'): # T15: epsilon = 0.2
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                # print("para.data_intial!!!!:", self.backup[name])
                norm = th.norm(param.grad)
                if norm != 0 and not th.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    # print("r_at!!!!!:", r_at)
                    param.data.add_(r_at)
                    # print("para.data!!!!:", param.data)

    def restore(self, emb_name='hard_fc1.'):

        for name, param in self.model.named_parameters():
            # print("param.requires_grad!!!!!!!!!!!", param.requires_grad)
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    '''
    Example
    pgd = PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
    K = 3
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        pgd.backup_grad()
        # 对抗训练
        for t in range(K):
            pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''

    def __init__(self, model, emb_name='hard_fc1.', epsilon=0.1, alpha=0.2):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = th.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if th.norm(r) > epsilon:
            r = epsilon * r / th.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]



class GCN_Net(th.nn.Module): 
    def __init__(self,in_feats,hid_feats,out_feats): 
        super(GCN_Net, self).__init__() 
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        self.fc=th.nn.Linear(2*out_feats,4)
        self.hard_fc1 = hard_fc(out_feats, out_feats)
        self.hard_fc2 = hard_fc(out_feats, out_feats) # optional

    def forward(self, data):
        init_x0, init_x, tdedge_index1, buedge_index1,tdedge_index2, buedge_index2= data.x0, data.x,\
                                                                                    data.TDnew_edge_index, data.BUnew_edge_index, \
                                                                          data.TDnew_edge_index2, data.BUnew_edge_index2
        # print("x!!!!!!!!!!!!!!", init_x0, init_x0.shape)
        x1_td =  self.conv1(init_x0, tdedge_index1)
        x1_td = F.relu(x1_td)
        x1_td = F.dropout(x1_td, training=self.training)
        x1_td = self.conv2(x1_td, tdedge_index1)
        x1_td = F.relu(x1_td)
        x1_td = scatter_mean(x1_td, data.batch, dim=0)
        x1_td_g = x1_td
        x1_td = self.hard_fc1(x1_td)
        x1_td_t = x1_td
        x1_td = th.cat((x1_td_g, x1_td_t), 1)

        x1_bu = self.conv1(init_x0, buedge_index1)
        x1_bu = F.relu(x1_bu)
        x1_bu = F.dropout(x1_bu, training=self.training)
        x1_bu = self.conv2(x1_bu, buedge_index1)
        x1_bu = F.relu(x1_bu)
        x1_bu = scatter_mean(x1_bu, data.batch, dim=0)
        x1_bu_g = x1_bu
        x1_bu = self.hard_fc1(x1_bu)
        x1_bu_t = x1_bu
        x1_bu = th.cat((x1_bu_g, x1_bu_t), 1)
        x1 = x1_td + x1_bu
        # print("x1!!!!!!!!!!!!!!!",x1)

        x2_td = self.conv1(init_x,tdedge_index2)
        x2_td = F.relu(x2_td)
        x2_td = F.dropout(x2_td, training=self.training)
        x2_td = self.conv2(x2_td, tdedge_index2)
        x2_td = F.relu(x2_td)
        x2_td = scatter_mean(x2_td, data.batch, dim=0)
        x2_td_g = x2_td
        x2_td = self.hard_fc1(x2_td)
        x2_td_t = x2_td
        x2_td = th.cat((x2_td_g, x2_td_t), 1)

        x2_bu = self.conv1(init_x, buedge_index2)
        x2_bu = F.relu(x2_bu)
        x2_bu = F.dropout(x2_bu, training=self.training)
        x2_bu = self.conv2(x2_bu, buedge_index2)
        x2_bu = F.relu(x2_bu)
        x2_bu = scatter_mean(x2_bu, data.batch, dim=0)
        x2_bu_g = x2_bu
        x2_bu = self.hard_fc1(x2_bu)
        x2_bu_t = x2_bu
        x2_bu = th.cat((x2_bu_g, x2_bu_t), 1)
        x2 = x2_td + x2_bu


        x = th.cat((x1, x2), 0)
        y = th.cat((data.y1, data.y2), 0)

        x_T = x.t()
        dot_matrix = th.mm(x, x_T)
        x_norm = th.norm(x, p=2, dim=1)#范数
        x_norm = x_norm.unsqueeze(1)
        norm_matrix = th.mm(x_norm, x_norm.t())
        
        t = 0.3 # pheme: t = 0.6
        cos_matrix = (dot_matrix / norm_matrix) / t
        cos_matrix = th.exp(cos_matrix)
        diag = th.diag(cos_matrix)
        cos_matrix_diag = th.diag_embed(diag)
        cos_matrix = cos_matrix - cos_matrix_diag
        y_matrix_T = y.expand(len(y), len(y))
        y_matrix = y_matrix_T.t()
        y_matrix = th.ne(y_matrix, y_matrix_T).float()
        #y_matrix_list = y_matrix.chunk(3, dim=0)
        #y_matrix = y_matrix_list[0]
        neg_matrix = cos_matrix * y_matrix
        neg_matrix_list = neg_matrix.chunk(2, dim=0)
        #neg_matrix = neg_matrix_list[0]
        pos_y_matrix = y_matrix * (-1) + 1
        pos_matrix_list = (cos_matrix * pos_y_matrix).chunk(2,dim=0)
        #print('cos_matrix: ', cos_matrix.shape, cos_matrix)
        #print('pos_y_matrix: ', pos_y_matrix.shape, pos_y_matrix)
        pos_matrix = pos_matrix_list[0]
        #print('pos shape: ', pos_matrix.shape, pos_matrix)
        neg_matrix = (th.sum(neg_matrix, dim=1)).unsqueeze(1)
        sum_neg_matrix_list = neg_matrix.chunk(2, dim=0)
        p1_neg_matrix = sum_neg_matrix_list[0]
        p2_neg_matrix = sum_neg_matrix_list[1]
        neg_matrix = p1_neg_matrix
        #print('neg shape: ', neg_matrix.shape)
        div = pos_matrix / neg_matrix 
        div = (th.sum(div, dim=1)).unsqueeze(1)  
        div = div / batchsize
        log = th.log(div)
        SUM = th.sum(log)
        cl_loss = -SUM

        x = self.fc(x) 
        x = F.log_softmax(x, dim=1)

        return x, cl_loss, y


class GCN_ATK(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(GCN_ATK, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        # self.fc = th.nn.Linear(out_feats, 4)

    def forward(self, data):
        init_x0, init_x, tdedge_index1, buedge_index1, tdedge_index2, buedge_index2 = data.x0, data.x, \
                                                                                      data.TDnew_edge_index, data.BUnew_edge_index, \
                                                                                      data.TDnew_edge_index2, data.BUnew_edge_index2

        x1_td = self.conv1(init_x0, tdedge_index1)
        x1_td = F.relu(x1_td)
        x1_td = F.dropout(x1_td, training=self.training)
        x1_td = self.conv2(x1_td, tdedge_index1)
        x1_td = F.relu(x1_td)
        x1_td = scatter_mean(x1_td, data.batch, dim=0)


        x1_bu = self.conv1(init_x0, buedge_index1)
        x1_bu = F.relu(x1_bu)
        x1_bu = F.dropout(x1_bu, training=self.training)
        x1_bu = self.conv2(x1_bu, buedge_index1)
        x1_bu = F.relu(x1_bu)
        x1_bu = scatter_mean(x1_bu, data.batch, dim=0)

        x1 = x1_td + x1_bu

        x2_td = self.conv1(init_x, tdedge_index2)
        x2_td = F.relu(x2_td)
        x2_td = F.dropout(x2_td, training=self.training)
        x2_td = self.conv2(x2_td, tdedge_index2)
        x2_td = F.relu(x2_td)
        x2_td = scatter_mean(x2_td, data.batch, dim=0)


        x2_bu = self.conv1(init_x, buedge_index2)
        x2_bu = F.relu(x2_bu)
        x2_bu = F.dropout(x2_bu, training=self.training)
        x2_bu = self.conv2(x2_bu, buedge_index2)
        x2_bu = F.relu(x2_bu)
        x2_bu = scatter_mean(x2_bu, data.batch, dim=0)
        x2 = x2_td + x2_bu

        x = th.cat((x1, x2), 0)
        y = th.cat((data.y1, data.y2), 0)

        x_T = x.t()
        dot_matrix = th.mm(x, x_T)
        x_norm = th.norm(x, p=2, dim=1)  # 范数
        x_norm = x_norm.unsqueeze(1)
        norm_matrix = th.mm(x_norm, x_norm.t())

        t = 0.3  # pheme: t = 0.6
        cos_matrix = (dot_matrix / norm_matrix) / t
        cos_matrix = th.exp(cos_matrix)
        diag = th.diag(cos_matrix)
        cos_matrix_diag = th.diag_embed(diag)
        cos_matrix = cos_matrix - cos_matrix_diag
        y_matrix_T = y.expand(len(y), len(y))
        y_matrix = y_matrix_T.t()
        y_matrix = th.ne(y_matrix, y_matrix_T).float()

        neg_matrix = cos_matrix * y_matrix
        neg_matrix_list = neg_matrix.chunk(2, dim=0)

        pos_y_matrix = y_matrix * (-1) + 1
        pos_matrix_list = (cos_matrix * pos_y_matrix).chunk(2, dim=0)

        pos_matrix = pos_matrix_list[0]

        neg_matrix = (th.sum(neg_matrix, dim=1)).unsqueeze(1)
        sum_neg_matrix_list = neg_matrix.chunk(2, dim=0)
        p1_neg_matrix = sum_neg_matrix_list[0]
        p2_neg_matrix = sum_neg_matrix_list[1]
        neg_matrix = p1_neg_matrix

        div = pos_matrix / neg_matrix
        div = (th.sum(div, dim=1)).unsqueeze(1)
        div = div / batchsize
        log = th.log(div)
        SUM = th.sum(log)
        cl_loss = -SUM

        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x, cl_loss, y


def train_GCN(x_test, x_train,lr, weight_decay,patience,n_epochs,batchsize,dataname):

    # print("x_train!!!!!!!!!!!!!!!!", x_train)
    model = GCN_Net(768,64,64).to(device)
    model_atk = GCN_ATK(768,64,64).to(device)
    fgm = FGM(model)
    pgd = PGD(model)
    fgm_atk = SINGLE_ATTACK(model_atk)



    for para in model.hard_fc1.parameters():
        para.requires_grad = False
    for para in model.hard_fc2.parameters():
        para.requires_grad = False
    #filter根据判断结果自动过滤掉不符合条件的元素，并返回有符合元素组成的新列表
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    #先最小化损失更新hard_fc1模型外其他的参数


    for para in model.hard_fc1.parameters():
        para.requires_grad = True
    for para in model.hard_fc2.parameters():
        para.requires_grad = True
    # #optimizer_hard = th.optim.Adam(model.hard_fc.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_hard = th.optim.SGD([{'params': model.hard_fc1.parameters()},
                                    {'params': model.hard_fc2.parameters()}], lr=0.001)
    #然后最大化损失更新hard_fc1的参数

    model.train() 
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True) 

    for epoch in range(n_epochs): 
        traindata_list, testdata_list = loadData(dataname, x_train, x_test, droprate=0.3) # T15 droprate = 0.1
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=0)
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        NUM=1
        beta=0.01
        K=3
        flag = 0

        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels, cl_loss, y = model(Batch_data) 
            finalloss = F.nll_loss(out_labels,y)
            loss = finalloss + beta*cl_loss
            avg_loss.append(loss.item())

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            ##------------- FGM ---------------##

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            fgm.attack()
            out_labels, cl_loss, y = model(Batch_data)
            finalloss = F.nll_loss(out_labels,y)
            loss_adv = finalloss + beta*cl_loss

            optimizer_hard.zero_grad()
            loss_adv.backward()
            fgm.restore()
            optimizer_hard.step()

            ##--------------------------------##

            ##------------- S2PGD ---------------##
            '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t==0))
                if t != K-1:
                    optimizer_hard.zero_grad()
                else:
                    pgd.restore_grad()
                out_labels, cl_loss, y = model(Batch_data)
                finalloss = F.nll_loss(out_labels,y)
                loss_adv = finalloss + beta*cl_loss
                loss_adv.backward()

            pgd.restore()
            optimizer.step()
            '''
            ##-------------————————---------------##



            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y).sum().item()
            train_acc = correct / len(y) 
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,loss.item(),train_acc))
            batch_idx = batch_idx + 1
            NUM += 1
            #print('train_loss: ', loss.item())

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []


        tqdm_test_loader = tqdm(test_loader)
        ##——————————测试——————————##
        model.eval()
        # tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            # print("Batch_data.x!!!!!!!!!!!!!!!!!!", Batch_data.x)
            fgm_atk.attack(Batch_data, beta)
            # print("RETURN_Batch_data.x!!!!!!!!!!!!!!!!!!", Batch_data.x)

            val_out, val_cl_loss, y = model(Batch_data)
            val_loss = F.nll_loss(val_out, y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(y).sum().item()
            val_acc = correct / len(y) 
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, y) 
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Avg_Loss {:.4f} | Val_Losses {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(avg_loss),
                                                                            np.mean(temp_val_losses), np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
      
        if epoch > 25:
            early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                        np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'GACL', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return accs,F1,F2,F3,F4


##---------------------------------main---------------------------------------
scale = 1
lr=0.0005 * scale
weight_decay=1e-4
patience=10
n_epochs=200
batchsize=120  
datasetname='Twitter16' # (1)Twitter15  (2)pheme  (3)weibo
#model="GCN" 
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = [] 
NR_F1 = [] # NR
FR_F1 = [] # FR
TR_F1 = [] # TR
UR_F1 = [] # UR

# data_path = './data/twitter15/'
# label_path = './data/Twitter15_label_All.txt'

data_path = './data/twitter16/'
label_path = './data/Twitter16_label_All.txt'

fold0_x_test, fold0_x_train, \
fold1_x_test,  fold1_x_train,\
fold2_x_test, fold2_x_train, \
fold3_x_test, fold3_x_train, \
fold4_x_test,fold4_x_train = load5foldData(datasetname,data_path,label_path)

print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train))
print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train))
print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train))
print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train))
print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train))


accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(fold0_x_test,fold0_x_train,lr,weight_decay, patience,n_epochs,batchsize,datasetname)
accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(fold1_x_test,fold1_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(fold2_x_test,fold2_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(fold3_x_test,fold3_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(fold4_x_test,fold4_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)


test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5) 
FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5) 
TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5) 
UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
print("AVG_result: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(sum(test_accs), sum(NR_F1), sum(FR_F1), sum(TR_F1), sum(UR_F1)))


time_end = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print("time_end!!!!!!!!!!!!!!!!", time_end)