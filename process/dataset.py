import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
#from transformers import *
import json
from torch.utils.data import DataLoader
from torch_geometric.utils import degree, to_undirected
from torch_scatter import scatter


# global
label4id = {
            "unverified": 0,
            "non-rumor": 1,
            "true": 2,
            "false": 3,
            }

label2id = {
            "true": 0,
            "false": 1,
            }

def random_pick(list, probabilities): 
    x = random.uniform(0,1)
    cumulative_probability = 0.0 
    for item, item_probability in zip(list, probabilities): 
         cumulative_probability += item_probability 
         if x < cumulative_probability:
               break 
    return item 

class RumorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])#torch.LongTensor(batch_label).to(device1)
        #item['labels'] = torch.LongTensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def edge_weight(deg_col, p, yu):

    s_col = torch.log(deg_col)

    if s_col.max().equal(s_col.mean()):
        weights = s_col.max() - s_col.mean() / torch.ones(len(s_col))
        edge_weights = weights
    else:
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
        edge_weights = weights
        edge_weights = edge_weights / edge_weights.mean() * p

    # print("edge_weights!!!!!!!!!!!!!!", edge_weights)
    edge_weights = edge_weights.where(edge_weights < yu, torch.ones_like(edge_weights) * yu)

    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    return sel_mask


def feature_drop_weights_dense(x, deg_x, p, yu):

    u = x.clone()
    x = x.abs()

    if x.shape[0] == deg_x.shape[0]:
        w = x.t() @ deg_x
        w = w.log()
        s = (w.max() - w) / (w.max() - w.mean())

        s = s / s.mean() * p
        # print("s!!!!!!!!!", s)
        s = s.where(s < yu, torch.ones_like(s) * yu)
        drop_prob = s
        # print("drop_prob!!!!!!!!!!!!", drop_prob)
        drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
        u[:, drop_mask] = 0.

    return u


class GraphDataset(Dataset):
    def __init__(self, fold_x, droprate): 
        
        self.fold_x = fold_x
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index]
    
        # ====================================edgeindex========================================
        with open('./data/twitter16/' + id + '/after_tweets.pkl', 'rb') as t:
        # with open('./data/twitter15/'+ id + '/after_tweets.pkl', 'rb') as t:
        # with open('./data/all/' + id + '/tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        # print("tweets!!!!!!!!!!!!!!:",tweets)
        # print("train_id!!!!!!!!!!!!!!!!", id)

        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        # print('dict!!!!!!!!!!!!!!: ', dict)

        with open('./data/twitter16/' + id + '/after_structure.pkl', 'rb') as f:
        # with open('./data/twitter15/'+ id + '/after_structure.pkl', 'rb') as f:
        # with open('./data/all/' + id + '/structure.pkl', 'rb') as f:
            inf = pickle.load(f)
            # print(id)
        # print("inf!!!!!!!!!!!!", len(inf), inf)


        inf = inf[1:]
        new_inf = []

        ########early detection
        # max = 0
        # new_id = id + '.txt'
        # list_2 = []
        # list_time = []
        #
        # # for line_2 in open('./raw/twitter16/last_tree/' + new_id):
        # for line_2 in open('./raw/twitter15/last_tree/' + new_id):
        #     dict_xin_1, dict_xin_2, time_2 = line_2.split('\t')[0].strip(" "), \
        #                                      line_2.split('\t')[1].strip(" "), line_2.split('\t')[2].strip(" ")
        #     dict_2 = [dict_xin_1, dict_xin_2]
        #     list_2.append(dict_2)
        #     time_2 = float(time_2)
        #     list_time.append(time_2)
        #
        # ########early detection
        # # for pair in inf:
        # #     for i in range(len(list_2)):
        # #         if (pair == list_2[i] and list_time[i]<=10.0):
        # #             new_pair = []
        # #             for E in pair:
        # #                 if E == 'ROOT':
        # #                     break
        # #                 E = dict[E]
        # #                 if( E>max ):
        # #                     max = E
        # #                 new_pair.append(E)
        # #
        # #             if E != 'ROOT':
        # #                 new_inf.append(new_pair)
        # #                 break
        #
        # ######没有边的情况
        # if(len(new_inf)==0):
        #     list_min = min(list_time)
        #     for i in range(len(list_time)):
        #         if (list_time[i] <= list_min):
        #             pair = list_2[i]
        #             new_pair = []
        #             for E in pair:
        #                 if E == 'ROOT':
        #                     break
        #                 E = dict[E]
        #                 if (E > max):
        #                     max = E
        #                 new_pair.append(E)
        #
        #             if E != 'ROOT':
        #                 new_inf.append(new_pair)
        #                 break
        #         if(max != 0):
        #             break
        #
        #
        # max += 1


        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)


        new_inf = np.array(new_inf).T
        edgeindex = new_inf

        yu = self.droprate
        p1 = 0.6  # 0.15  #0.6
        p2 = 0.4  # 0.2  #0.4

        init_row = list(edgeindex[0]) 
        init_col = list(edgeindex[1])
        # print('init_row_xin!!!!!!!!!!!!!!: ', len(init_row), "\n", init_row)
        # print('init_col_xin!!!!!!!!!!!!!!: ', len(init_col), "\n", init_col)

        burow = list(edgeindex[1]) 
        bucol = list(edgeindex[0])

        row = init_row + burow
        col = init_col + bucol
        a = []
        b = []
        alln = 0


        #==================================- dropping + adding + misplacing -===================================#

        choose_list = [1,2,3] # 1-drop 2-add 3-misplace
        probabilities = [0.3,0,0.7] # T15: probabilities = [0.5,0.3,0.2]
        choose_num = random_pick(choose_list, probabilities)



        tdnew_edgeindex = [init_row, init_col]
        # print("tdnew_edgeindex!!!!!!!!!", torch.LongTensor(tdnew_edgeindex).shape)

        if self.droprate > 0 and len(edgeindex[1]) >2  :
            # row = list(edgeindex[0])
            # col = list(edgeindex[1])
            # row_1 = torch.tensor(edgeindex[0]).to(torch.long)
            # col_1 = torch.tensor(edgeindex[1]).to(torch.long)


            if choose_num == 1:

                row_2 = torch.tensor(edgeindex).to(torch.long)
                # print("edge_index!!!!!!!!1",edge_index_)
                # deg = degree(edge_index_[1])
                deg = degree(row_2[1])
                deg_col = deg[edgeindex[1]].to(torch.float32)
                sel_mask = edge_weight(deg_col, p1, yu)
                # print("sel_mask!!!!!!!!!!!!!", type(sel_mask), sel_mask)
                tdnew_edgeindex2 = edgeindex[:, sel_mask]
                # print("tdnew_edgeindex2!!!!!!!!!!!!", tdnew_edgeindex2)
                         
            elif choose_num == 3:
                k=0
                row_2 = torch.tensor(edgeindex).to(torch.long)
                # edge_index_ = to_undirected(row_2)
                # deg = degree(edge_index_[1])
                deg = degree(row_2[1])
                deg_col = deg[edgeindex[1]].to(torch.float32)
                sel_mask = edge_weight(deg_col, p1, yu)
                # print("sel_mask!!!!!!!!!!!!!", type(sel_mask), sel_mask)

                for i in range(len(sel_mask)):
                    if sel_mask[i] ==False:
                        a.append(i)
                        k=k+1
                        alln= k
                        b.insert(i,1)
                k=0
                tdnew_edgeindex2 = edgeindex[:, sel_mask].tolist()
                for i in range(len(sel_mask)) :
                    if sel_mask[i] == False and b[k] ==1 and k<=float((alln-1)/2) :
                        if k != float((alln-1)/2):
                            top = a[k]
                            bottom =a[alln-k-1]
                            tdnew_edgeindex2[0].append(edgeindex[0][i])
                            tdnew_edgeindex2[1].append(edgeindex[1][bottom])
                            tdnew_edgeindex2[0].append(edgeindex[0][top])
                            tdnew_edgeindex2[1].append(edgeindex[1][i])
                        else:
                            top = a[k]
                            bottom = a[alln - k - 1]
                            tdnew_edgeindex2[0].append(edgeindex[0][i])
                            tdnew_edgeindex2[1].append(edgeindex[1][bottom])

            # else:
            #     length = len(row)
            #     poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            #     poslist = sorted(poslist)
            #     row2 = list(np.array(row)[poslist])
            #     col2 = list(np.array(col)[poslist])
            #     tdnew_edgeindex2 = [row2, col2]
        else:
             tdnew_edgeindex2 = [init_row , init_col]

        bunew_edgeindex = [burow, bucol]

        if self.droprate > 0 and len(edgeindex[1])>2:
            if choose_num == 1:
                row_2 = torch.tensor(edgeindex).to(torch.long)
                # edge_index_ = to_undirected(row_2)
                # deg = degree(edge_index_[0])
                deg = degree(row_2[0])
                deg_col = deg[edgeindex[0]].to(torch.float32)
                sel_mask = edge_weight(deg_col, p1, yu)
                bunew_edgeindex2 = edgeindex[:, sel_mask]


            elif choose_num == 3:
                k = 0
                row_2 = torch.tensor(edgeindex).to(torch.long)
                # edge_index_ = to_undirected(row_2)
                # deg = degree(edge_index_[0])
                deg = degree(row_2[0])
                deg_col = deg[edgeindex[0]].to(torch.float32)
                sel_mask = edge_weight(deg_col, p1, yu)
                bunew_edgeindex2 = edgeindex[:, sel_mask]


                for i in range(len(sel_mask)):
                    if sel_mask[i] == False:
                        a.append(i)
                        k = k + 1
                        alln = k
                        b.insert(i, 1)
                k = 0
                bunew_edgeindex2 = edgeindex[:, sel_mask].tolist()
                for i in range(len(sel_mask)):
                    if sel_mask[i] == False and b[k] == 1 and k <= float((alln - 1) / 2):
                        if k != float((alln - 1) / 2):
                            top = a[k]
                            bottom = a[alln - k - 1]
                            bunew_edgeindex2[0].append(edgeindex[0][i])
                            bunew_edgeindex2[1].append(edgeindex[1][bottom])
                            bunew_edgeindex2[0].append(edgeindex[0][top])
                            bunew_edgeindex2[1].append(edgeindex[1][i])
                        else:
                            top = a[k]
                            bottom = a[alln - k - 1]
                            bunew_edgeindex2[0].append(edgeindex[0][i])
                            bunew_edgeindex2[1].append(edgeindex[1][bottom])
            # else:
            #     length = len(burow)
            #     poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            #     poslist = sorted(poslist)
            #     row2 = list(np.array(burow)[poslist])
            #     col2 = list(np.array(bucol)[poslist])
            #     bunew_edgeindex2 = [row2, col2]

        else:
            bunew_edgeindex2 = [burow, bucol]


        
        # =========================================X===============================================
        with open('./bert_w2c/T16/t16_mask_00/' + id + '.json', 'r') as j_f0:
        # with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f0:
        # with open('./bert_w2c/pheme/pheme_mask/' + id + '.json', 'r') as j_f0:
            json_inf0 = json.load(j_f0)


        x0_list  = json_inf0[id]

        #######根特征增强
        # x0_list_root = []
        # for i in range(len(x0_list)):
        #     x0_list_root.append(x0_list[0])

        # ####early detection
        # x0_list = x0_list[:max]

        x0 = np.array(x0_list)

        #######根特征增强
        # x0_root = np.array(x0_list_root)
        # x0 = np.concatenate((x0, x0_root), axis=1)

        # print("x0!!!!!!!!!!!!!!!!!!!!!", x0.shape)
        x0 = torch.tensor(x0,dtype=torch.float32)

        with open('./bert_w2c/T16/t16_mask_015/' + id + '.json', 'r') as j_f:
        # with open('./bert_w2c/T15/t15_mask_015/' + id + '.json', 'r') as j_f:
        # with open('./bert_w2c/pheme/pheme_mask/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)
        
        x_list = json_inf[id]

        #######根特征增强
        # x_list_root = []
        # for i in range(len(x_list)):
        #     x_list_root.append(x_list[0])

        # ######early detection
        # x_list = x_list[:max]


        x = np.array(x_list)


        ######根特征增强
        # x_root = np.array(x_list_root)
        # x = np.concatenate((x, x_root), axis=1)


        x = torch.tensor(x,dtype=torch.float32)
        # print("x!!!!!!!!!!!!!", x)

        # with open('./data/label_pheme.json', 'r') as j_tags:
        # with open('./data/label_15.json', 'r') as j_tags:
        with open('./data/label_16.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        # y = label4id[tags[id]]
        # y = np.array(y)
        if self.droprate > 0:

            row_x = torch.tensor(edgeindex).to(torch.long)
            edge_index_x = to_undirected(row_x)
            deg_x = degree(edge_index_x[1])
            x = feature_drop_weights_dense(x, deg_x, p2, yu)
            x0 = feature_drop_weights_dense(x0, deg_x, p2, yu)



            '''
            if choose_num == 1:
                zero_list = [0]*768
                x_length = len(x_list)
                r_list = random.sample(range(x_length), int(x_length * self.droprate))
                r_list = sorted(r_list)
                for idex, line in enumerate(x_list):
                    for r in r_list:
                        if idex == r:
                            x_list[idex] = zero_list

                x2 = np.array(x_list)
                x = x2
            '''
        # print("x0!!!!!!!!!!!!!!!!!!", x0.shape, torch.tensor(x0,dtype=torch.float32).shape)
        # print("tdnew_edgeindex!!!!!!!!!!!!", bunew_edgeindex)

        # print("Shape!!!!",torch.LongTensor(tdnew_edgeindex).shape, "Shape_x!!!!!",x0.shape)
        return Data(x0=x0,
                    x=x,
                    TDnew_edge_index = torch.LongTensor(tdnew_edgeindex),
                    BUnew_edge_index = torch.LongTensor(bunew_edgeindex),
                    TDnew_edge_index2 = torch.LongTensor(tdnew_edgeindex2),
                    BUnew_edge_index2 = torch.LongTensor(bunew_edgeindex2),
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y]))




class test_GraphDataset(Dataset):
    def __init__(self, fold_x, droprate): 
        
        self.fold_x = fold_x
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index): 
        id =self.fold_x[index] 
        # ====================================edgeindex==============================================
        with open('./data/twitter16/' + id + '/after_tweets.pkl', 'rb') as t:
        # with open('./data/twitter15/'+ id + '/after_tweets.pkl', 'rb') as t:
        # with open('./data/all/' + id + '/tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
            # print(id)
        #print(tweets)
        # print("test_id!!!!!!!!!!!!!!!!", id)

        
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index
        #print('dict: ', dict)

        with open('./data/twitter16/' + id + '/after_structure.pkl', 'rb') as f:
        # with open('./data/twitter15/'+ id + '/after_structure.pkl', 'rb') as f:
        # with open('./data/all/' + id + '/structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        inf = inf[1:]
        new_inf = []

        # ################early detection
        # max_test = 0
        # new_id = id + '.txt'
        # list_2 = []
        # list_time = []
        #
        # # for line_2 in open('./raw/twitter16/last_tree/' + new_id):
        # for line_2 in open('./raw/twitter15/last_tree/' + new_id):
        #     dict_xin_1, dict_xin_2, time_2 = line_2.split('\t')[0].strip(" "), \
        #                                      line_2.split('\t')[1].strip(" "), line_2.split('\t')[2].strip(" ")
        #     dict_2 = [dict_xin_1, dict_xin_2]
        #     list_2.append(dict_2)
        #     time_2 = float(time_2)
        #     list_time.append(time_2)
        #
        # #########early detection
        # # for pair in inf:
        # #     for i in range(len(list_2)):
        # #         if (pair == list_2[i] and list_time[i] <= 10.0):
        # #             new_pair = []
        # #             for E in pair:
        # #                 if E == 'ROOT':
        # #                     break
        # #                 E = dict[E]
        # #                 if (E > max_test):
        # #                     max_test = E
        # #                 new_pair.append(E)
        # #
        # #             if E != 'ROOT':
        # #                 new_inf.append(new_pair)
        # #                 break
        #
        #
        # #########没有边的情况
        # if (len(new_inf) == 0):
        #     list_min = min(list_time)
        #     for i in range(len(list_time)):
        #         if (list_time[i] <= list_min):
        #             pair = list_2[i]
        #             new_pair = []
        #             for E in pair:
        #                 if E == 'ROOT':
        #                     break
        #                 E = dict[E]
        #                 if (E > max_test):
        #                     max_test = E
        #                 new_pair.append(E)
        #
        #             if E != 'ROOT':
        #                 new_inf.append(new_pair)
        #                 break
        #         if (max_test != 0):
        #             break
        #
        #
        # max_test += 1


        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)

        new_inf = np.array(new_inf).T
        edgeindex = new_inf
        
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        # row.extend(burow)
        # col.extend(bucol)

        tdnew_edgeindex = [row, col]
        tdnew_edgeindex2 = [row, col]
        bunew_edgeindex = [burow, bucol]
        bunew_edgeindex2 = [burow, bucol]


        # =========================================X====================================================
        with open('./bert_w2c/T16/t16_mask_00/' + id + '.json', 'r') as j_f:
        # with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f:
        # with open('./bert_w2c/pheme/pheme_mask/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)
            # print(id)
        
        x_test = json_inf[id]

        x_test_root = []
        for i in range(len(x_test)):
            x_test_root.append(x_test[0])

        # ####early detection
        # x_test = x_test[:max_test]
        x_test = np.array(x_test)

        #######根特征增强
        # x_test_root = np.array(x_test_root)
        # x_test = np.concatenate((x_test, x_test_root), axis=1)

        ########随机扰动
        # print("x!!!!!!!!!!!!!!!!", x.shape)
        # a = np.random.randint(0,x.shape[0])
        # b = np.random.random(100)
        # x[a] = x[a] * np.max(b)

        '''
                    if choose_num == 1:
                        zero_list = [0]*768
                        x_length = len(x_list)
                        r_list = random.sample(range(x_length), int(x_length * 0.3))
                        r_list = sorted(r_list)
                        for idex, line in enumerate(x_list):
                            for r in r_list:
                                if idex == r:
                                    x_list[idex] = zero_list

                        x2 = np.array(x_list)
                        x = x2
                    '''

        # with open('./data/label_pheme.json', 'r') as j_tags:
        # with open('./data/label_15.json', 'r') as j_tags:
        with open('./data/label_16.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        print(y)
        # y = label4id[tags[id]]
        #y = np.array(y)


        return Data(x0=torch.tensor(x_test,dtype=torch.float32),
                    x=torch.tensor(x_test,dtype=torch.float32),
                    TDnew_edge_index=torch.LongTensor(tdnew_edgeindex),
                    BUnew_edge_index=torch.LongTensor(bunew_edgeindex),
                    TDnew_edge_index2=torch.LongTensor(tdnew_edgeindex2),
                    BUnew_edge_index2=torch.LongTensor(bunew_edgeindex2),
                    y1=torch.LongTensor([y]),
                    y2=torch.LongTensor([y])) 
