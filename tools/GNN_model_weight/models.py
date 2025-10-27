import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.nn import global_mean_pool, EdgeConv, GATConv, GINConv, PNAConv
from torch.autograd import Function
from torch.distributions import Categorical

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 64),
                                  nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                  nn.ReLU(), nn.Linear(128, 128),nn.ReLU(), nn.Linear(128, 128)),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(256,256,),
                                  nn.ReLU(), nn.Linear(256, 256),nn.ReLU(), nn.Linear(256, 256)),aggr='add')
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(256, 64)
        self.lin3 = torch.nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.1)
        x = self.lin2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin3(x)
        #print(x.shape)
        return F.sigmoid(x)

# class LundNet(torch.nn.Module):
#     def __init__(self):
#         super(LundNet, self).__init__()
#         self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.ReLU(),
#                                             nn.Linear(32, 32), nn.ReLU()),aggr='add')
#         self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
#                                             nn.Linear(32, 32), nn.ReLU()),aggr='add')
#         self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),  nn.ReLU(),
#                                             nn.Linear(64, 64),  nn.ReLU()),aggr='add')
#         self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),  nn.ReLU(),
#                                             nn.Linear(64, 64),  nn.ReLU()),aggr='add')
#         self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),  nn.ReLU(),
#                                             nn.Linear(128, 128),  nn.ReLU()),aggr='add')
#         self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),  nn.ReLU(),
#                                             nn.Linear(128, 128),  nn.ReLU()),aggr='add')

#         self.seq1 = nn.Sequential(nn.Linear(448, 384),
#                                 nn.BatchNorm1d(num_features=384),
#                                 nn.ReLU())
#         # self.seq2 = nn.Sequential(nn.Linear(384, 256),
#         #                           nn.ReLU())
#         self.seq2 = nn.Sequential(nn.Linear(385, 256),
#                                   nn.ReLU())
#         self.lin = nn.Linear(256, 1)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         Ntrk = data.Ntrk
#         Ntrk = torch.unsqueeze(Ntrk, 1)
#         x1 = self.conv1(x, edge_index)
#         x2 = self.conv2(x1, edge_index)
#         x3 = self.conv3(x2, edge_index)
#         x4 = self.conv4(x3, edge_index)
#         x5 = self.conv5(x4, edge_index)
#         x6 = self.conv6(x5, edge_index)
#         x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
#         x = self.seq1(x)
#         x = global_mean_pool(x, batch)
#         x = torch.cat( (x, Ntrk) ,dim=1)
#         x = self.seq2(x)
#         x = F.dropout(x, p=0.1)
#         x = self.lin(x)
#         #print(x.shape)
#         return F.sigmoid(x)

class LundNet4Class(torch.nn.Module):
    def __init__(self):
        super(LundNet4Class, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        # self.seq2 = nn.Sequential(nn.Linear(384, 256),
        #                           nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(385, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 4)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Ntrk = data.Ntrk
        Ntrk = torch.unsqueeze(Ntrk, 1)
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = torch.cat( (x, Ntrk) ,dim=1)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        return F.log_softmax(x, dim=1) # For multiclass classification
        # return x # Return length 4 output vector for use with nn.CrossEntropyLoss()


class LundNet(torch.nn.Module):
    def __init__(self):
        super(LundNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        # self.seq2 = nn.Sequential(nn.Linear(384, 256),
        #                           nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(385, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Ntrk = data.Ntrk
        Ntrk = torch.unsqueeze(Ntrk, 1)
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = torch.cat( (x, Ntrk) ,dim=1)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class LundNet_plus_GN2X(torch.nn.Module):
    def __init__(self):
        super(LundNet_plus_GN2X, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(num_features=32), nn.ReLU(),
                                            nn.Linear(32, 32), nn.BatchNorm1d(num_features=32), nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(num_features=64), nn.ReLU(),
                                            nn.Linear(64, 64), nn.BatchNorm1d(num_features=64), nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(num_features=128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.BatchNorm1d(num_features=128), nn.ReLU()),aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        # self.seq2 = nn.Sequential(nn.Linear(384, 256),
        #                           nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(389, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Ntrk = data.Ntrk
        Ntrk = torch.unsqueeze(Ntrk, 1)
        
        GN2X_pqcd = data.GN2X_pqcd
        GN2X_pqcd = torch.unsqueeze(GN2X_pqcd, 1)
        GN2X_phbb = data.GN2X_phbb
        GN2X_phbb = torch.unsqueeze(GN2X_phbb, 1)
        GN2X_ptop = data.GN2X_ptop
        GN2X_ptop = torch.unsqueeze(GN2X_ptop, 1)
        GN2X_phcc = data.GN2X_phcc
        GN2X_phcc = torch.unsqueeze(GN2X_phcc, 1)
        
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        #x = torch.cat( (x, Ntrk) ,dim=1)
        x = torch.cat( (x, Ntrk, GN2X_pqcd, GN2X_phbb, GN2X_ptop, GN2X_phcc) ,dim=1)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class LundNet_old(torch.nn.Module):
    def __init__(self):
        super(LundNet_old, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        '''
        print("x1->",x1)
        print( len(x1[0]))
        print("x2->",x2)
        print( len(x2[0]))
        print("x3->",x3)
        print( len(x3[0]))
        print("x4->",x4)
        print( len(x4[0]))
        print("x5->",x5)
        print( len(x5[0]))
        print("x6->",x6)
        print( len(x6[0]))
        print( len(x[30]) )
        print( len(x) )
        print(x[:2])
        '''
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class GATNet(torch.nn.Module):
    def __init__(self):
        in_channels = 3
        super(GATNet, self).__init__()
        self.conv1 = GATConv(4, 16, heads=8, dropout=0.1)
        self.conv2 = GATConv(16 * 8, 16, heads=8, dropout=0.1)
        self.conv3 = GATConv(16 * 8, 32, heads=8, dropout=0.1)
        self.conv4 = GATConv(32 * 8, 32, heads=16, dropout=0.1)
        self.conv5 = GATConv(16 * 32, 64, heads=16, dropout=0.1)
        self.conv6 = GATConv(64 * 16, 64, heads=16, dropout=0.1)
        self.seq1 = nn.Sequential(nn.Linear(3072, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = torch.nn.Linear(256, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(3, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),train_eps=True)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),train_eps=True)
        self.conv3 = GINConv(nn.Sequential(nn.Linear(32,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv4 = GINConv(nn.Sequential(nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(64, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        self.conv6 = GINConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)

class EdgeGinNet(torch.nn.Module):
    def __init__(self):
        super(EdgeGinNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv4 = GINConv(nn.Sequential(nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(64, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        self.conv6 = GINConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)



class PNANet(torch.nn.Module):
    def __init__(self,in_channels):
        aggregators = ['sum','mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation',"linear",'inverse_linear']
        in_channels = 3
        super(PNANet, self).__init__()
        self.conv1 = PNAConv(in_channels, out_channels=32, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNAConv(in_channels=32, out_channels=64, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNAConv(in_channels=64, out_channels=128, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv4 = PNAConv(in_channels=128, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv5 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv6 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.seq1 = nn.Sequential(nn.Linear(992, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2,x3,x4,x5,x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        return F.sigmoid(x)

class ConcatDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        return xA, xB

    def __len__(self):
        return len(self.datasetA)



################################################################################
## from adv_script
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input

        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.bn1 = nn.BatchNorm1d(num_features=in_features)
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        #self.sigma = nn.Sequential(
        #    nn.Linear(in_features, out_features * num_gaussians),
        #    nn.Softplus()
        #)

        #self.mu = nn.Sequential(
        #    nn.Linear(in_features, out_features * num_gaussians),
        #    nn.Sigmoid()
        #)

        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        minibatch = self.bn1(minibatch)
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    ret = torch.where(ret == 0, ret + 1E-20, ret)
    return torch.prod(ret, 2)



def mdn_loss(pi, sigma, mu, target, weight):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """

    #print("pi->",pi.size() )
    #print("sigma->",sigma.size() )
    #print("mu->",mu.size() )
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -weight*torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


###################################################################----------------------------------------------------
class MDN_new(nn.Module):
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN_new, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.bn1 = nn.BatchNorm1d(num_features=in_features)
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features, out_features * num_gaussians),
            nn.Softplus()
        )

        self.mu = nn.Sequential(
            nn.Linear(in_features, out_features * num_gaussians),
            nn.Sigmoid()
        )
    def forward(self, minibatch):
        minibatch = self.bn1(minibatch)
        pi = self.pi(minibatch)
        #sigma = torch.exp(self.sigma(minibatch))
        sigma = self.sigma(minibatch)
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability_new(sigma, mu, target):
    target = (target - 40)/( 300 - 40 )
    #print("mass_0to1",target[:2])
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    ret = torch.where(ret == 0, ret + 1E-20, ret)
    return torch.prod(ret, 2)


def pi_redefinition(device, pi, sigma, mu):
    # redefine pi in order to obtain normalized distributions inside [0,1] interval
    # mu = Mean
    # sigma = width
    t_ones = torch.ones( mu.size() )
    t_zeros = torch.zeros( mu.size() )
    t_ones = t_ones.to(device)
    t_zeros = t_zeros.to(device)

    z0 = (t_zeros - mu) / sigma
    z1 = (t_ones - mu) / sigma
    out_interval = 0.5 * (1. + torch.erf(z1 / np.sqrt(2.)))  - 0.5 * (1. + torch.erf(z0 / np.sqrt(2.)))  # area inside [0,1]
    #print("pi size->",pi.size(),"   out_interval size->",out_interval.size() )
    pi_2 = pi / out_interval.view(pi.size())
    #print("pi_2[0]",pi_2[0])
    '''
    for i in range (0,len(pi_2) ): # sum all gaussians must be =1
        #pi_2[i] = pi_2[i]/torch.sum(pi_2[i])
        pi_2[i] /= torch.sum(pi_2[i])
    '''

    #print("pi size after:",pi_2.size() )
    #print(pi_2.size,"  ",pi_2[:2])
    return pi_2

def mdn_loss_new(device, pi, sigma, mu, target, weight):
    #print("pi---------------------------------------")
    #print(pi[:2])

    pi_2 = pi_redefinition(device, pi, sigma, mu)
    #pi_2 = pi

    ##redefine mu inside interval [-0.3,1.3] ~~ mu*1.6 - 0.3
    mu_shift = 0.3 * torch.ones( mu.size() )
    mu_shift = mu_shift.to(device)
    mu_2 = 1.6 * mu - mu_shift

    '''
    print("---------------------------------------")
    print("mass",target[:2])
    print("---------------------------------------")
    print("mu size->", mu.size(), "   pi size->",pi.size() ,"   sigma size->", sigma.size()  )
    print("mu_2---------------------------------------")
    print(mu_2[:2])
    print("pi_2---------------------------------------")
    print(pi_2[:2])
    print("sigma---------------------------------------")
    print(sigma[:2])
    print("---------------------------------------")
    '''
    prob = pi_2 * gaussian_probability_new(sigma, mu_2, target)
    nll = -weight*torch.log(torch.sum(prob, dim=1))
    #nll = weight*torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)

class Adversary_new(nn.Module):
    def __init__(self, lambda_parameter, num_gaussians):
        super(Adversary_new, self).__init__()
        self.gauss = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    #nn.LeakyReLU(),
    MDN_new(64, 1, num_gaussians)
)
        self.revgrad = GradientReversal(lambda_parameter)
        #print("lambda = {}".format(lambda_parameter))
    def forward(self, x):
        x = self.revgrad(x) # important hyperparameter, the scale,   # tells by how much the classifier is punished
        x = self.gauss(x)
        return x

###################################################################----------------------------------------------------

def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)


# The architecture of the adversary could be changed
class Adversary(nn.Module):
    def __init__(self, lambda_parameter, num_gaussians):
        super(Adversary, self).__init__()
        self.gauss = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    MDN(64, 1, num_gaussians)
)
        self.revgrad = GradientReversal(lambda_parameter)
        #print("lambda = {}".format(lambda_parameter))

    def forward(self, x):
        x = self.revgrad(x) # important hyperparameter, the scale,
                                     # tells by how much the classifier is punished
        x = self.gauss(x)
        return x
