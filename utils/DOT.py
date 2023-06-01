'''
#####################################################################################################################
Author      : Ce Ju 
Date        : 1st, Jun., 2023
Descriptions:  



The class DOT_LEM implements the DOT functionality described in the paper, encompasses several types of deep transfer algorithms, 
and provides parameters for three datasets. Below we explain the meaning of all hyperparameters:


1. DOT_type: Includes ’MDA‘, ’CDA‘, ’MDA/CDA’, and ‘JDOA’ methods. If something else is entered, only the most basic cross entropy 
is run.

2. architecture: Includes 'BiMap', 'Tensor', and 'Graph'.

'BiMap' indicates that there is only one BiMap Layer, and one ReEig where the number of inputs and outputs equals the number of 
channels.

'Tensor' represents 'Tensor-CSPNet'; 'Graph' represents 'Graph-CSPNet'. The latter two architectures remove the Riemannian BN Layer 
because the RieBN Layer involves parallel transmission operations, which occur under the AIRM metric.

3. dataset: Includes 'KU', 'BNCI2014001', and 'BNCI2015001'.


Specifically, when invoking the CDA method, pseudo labels are required. In our method, we use 'target_label' as the variable for 
this purpose.

#######################################################################################################################
'''

from sklearn.preprocessing import OneHotEncoder
import geoopt
import ot
import torch as th
import torch.nn as nn


from .modules import *


class DOT_LEM(nn.Module):

    def __init__(self, channel_num, DOT_type='CE', architecture='BiMap', dataset = 'BNCI2015001', P=None, MLP=False, device='cpu'):

        super(DOT_LEM, self).__init__()

        self.DOT_type     = DOT_type
        self.architecture = architecture
        self.dataset      = dataset 

        if self.dataset  == 'KU':
            self.classes          = 2
            self.dims             = [20, 20]
            self.frequency_bands  = 9
        elif self.dataset == 'BNCI2014001':
            self.classes          = 4
            self.dims             = [22, 22]
            self.frequency_bands  = 9
        elif self.dataset == 'BNCI2015001':
            self.classes          = 2
            self.dims             = [13, 13]
            self.frequency_bands  = 9
    

        self.channel_num  = channel_num
        self.P            = P
        self.MLP          = MLP
        self.device       = device
        self.encoder      = OneHotEncoder(sparse=False)

        self.CSP_Block    = self._make_CSP_block(len(self.dims)//2).double()
        self.LogEig       = LogEig().double()
        self.Classifier   = nn.Linear(self.frequency_bands*self.dims[-1]**2, self.classes).double()

        if self.architecture == 'Tensor':
            self.sharedNet = Tensor_CSPNet_Basic(channel_num = self.channel_num, mlp = self.MLP, dataset = self.dataset).to(self.device)
        elif self.architecture == 'Graph':
            self.sharedNet = Graph_CSPNet_Basic(channel_num = self.channel_num, P = self.P, mlp = self.MLP, dataset = self.dataset).to(self.device)
        elif self.architecture == 'BiMap':
            self.sharedNet = self.bimap_transformation
    
    def _make_CSP_block(self, layer_num):

        layers = []
        for i in range(layer_num):
          dim_in, dim_out = self.dims[2*i], self.dims[2*i+1]
          layers.append(BiMap(self.channel_num, dim_in, dim_out))
          layers.append(ReEig())

        return nn.Sequential(*layers).double()

    def bimap_transformation(self, x):

        window_num, band_num = x.shape[1], x.shape[2]

        x     = x.reshape(x.shape[0], window_num*band_num, x.shape[3], x.shape[4])
        x_csp = self.CSP_Block(x)
        x_log = self.LogEig(x_csp)
        y     = self.Classifier(x_log.reshape(x_log.shape[0], -1))

        return y, x_csp, x_log
    
    def marginal_distribution(self, source, target):

        loss = th.zeros(1)[0]

        batch_size_s = source.shape[0]
        batch_size_t = target.shape[0]

        for channel in range(self.channel_num):
            loss += th.linalg.matrix_norm(source[:, channel, :, :].sum(axis=0)/batch_size_s - target[:, channel, :, :].sum(axis=0)/batch_size_t, 'fro')**2

        return loss.sum()
    
    def conditional_distribution(self, source, source_label, target, target_label):

        loss = th.zeros(1)[0]

        for cl in range(self.classes):

          source_cl = source[[np.where(source_label == cl)[0]]]
          target_cl = target[[np.where(target_label == cl)[0]]]

          m = source_cl.data.shape[0]
          n = target_cl.data.shape[0]

          if m > 0 and n >0:
            for channel in range(self.channel_num):
              loss += th.linalg.matrix_norm(source_cl[:, channel, :, :].sum(axis=0)/m - target_cl[:, channel, :, :].sum(axis=0)/n, 'fro')**2

        return loss.sum()

    def category_vec(self, x):
        return th.from_numpy(self.encoder.fit_transform(x))

    def jdot(self, source, target, source_label, target_pred_logits):

        target_pred_label = th.argmax(F.log_softmax(target_pred_logits, dim = -1), dim=1)

        a = th.ones((len(source),), device=self.device)/len(source)
        b = th.ones((len(target),), device=self.device)/len(target)

        jdot_loss = th.zeros(1)[0]

        manifold_spd = geoopt.SymmetricPositiveDefinite("LEM")

        for channel in range(self.channel_num):

          source_channel = source[:, channel, :, :]
          target_channel = target[:, channel, :, :]

          M = manifold_spd.dist(source_channel[:,None], target_channel[None]) + th.cdist(self.category_vec(source_label.view(1, -1)), 
                                                                                             self.category_vec(target_pred_label.view(1, -1)), 
                                                                                             p=2)

          P = ot.emd(a, b, M**2)

          jdot_loss += th.sum(th.mul(P.double(),M.double()))

        return jdot_loss

    def forward(self, source, source_label, target, target_label):

        loss = th.zeros(1)[0]
        source_y, source_x_csp, source_x_log = self.sharedNet(source)

        if self.training == True:

            target_y, target_x_csp, target_x_log = self.sharedNet(target)

            if self.DOT_type == 'MDA' or 'MDA/CDA':
                loss += self.marginal_distribution(source_x_log, target_x_log)
            elif self.DOT_type == 'CDA' or 'MDA/CDA':
                loss += self.conditional_distribution(source_x_log, source_label, target_x_log, target_label)
            elif self.DOT_type == 'JDOT':
                loss += self.jdot(source_x_csp, target_x_csp, source_label, target_y)

        return source_y, loss


class Tensor_CSPNet_Basic(nn.Module):

    def __init__(self, channel_num, mlp, dataset = 'KU'):
        super(Tensor_CSPNet_Basic, self).__init__()

        self._mlp             = mlp
        self.channel_in       = channel_num

        if dataset == 'KU':
            classes           = 2
            self.dims         = [20, 30, 30, 20]
            self.kernel_size  = 3
            self.tcn_channles = 48
        elif dataset == 'BNCI2014001':
            classes           = 4
            self.dims         = [22, 36, 36, 22]
            self.kernel_size  = 2
            self.tcn_channles = 16
        elif dataset == 'BNCI2015001'
            classes           = 2
            self.dims         = [13, 30, 30, 13]
            self.kernel_size  = 5
            self.tcn_channeles = 32

        self.BiMap_Block      = self._make_bimap_block(len(self.dims)//2)
        self.LogEig           = LogEig()

        #The width of tcn is among 1 to 9, and 9 is the best usually. 
        self.tcn_width        =  9 

        self.Temporal_Block   = nn.Conv2d(1, self.tcn_channles, (self.kernel_size, self.tcn_width*self.dims[-1]**2), stride=(1, self.dims[-1]**2), padding=0).double()
        
        if self._mlp:
            self.Classifier = nn.Sequential(
            nn.Linear(self.tcn_channles, self.tcn_channles),
            nn.ReLU(inplace=True),
            nn.Linear(self.tcn_channles, self.tcn_channles),
            nn.ReLU(inplace=True),
            nn.Linear(self.tcn_channles, classes)
            ).double()
        else:
            self.Classifier = nn.Linear(self.tcn_channles, classes).double()
    
    def _make_bimap_block(self, layer_num):
        layers = []
        for i in range(layer_num):
            dim_in, dim_out = self.dims[2*i], self.dims[2*i+1]
            layers.append(BiMap(self.channel_in, dim_in, dim_out))
            layers.append(ReEig())
        return nn.Sequential(*layers).double()

    def forward(self, x):

        window_num, band_num = x.shape[1], x.shape[2]

        x     = x.reshape(x.shape[0], window_num*band_num, x.shape[3], x.shape[4])

        x_csp = self.BiMap_Block(x)

        x_log = self.LogEig(x_csp)

        # NCHW Format: (batch_size, window_num*band_num, 4, 4) ---> (batch_size, 1, window_num, band_num * 4 * 4)
        x_vec = x_log.view(x_log.shape[0], 1, window_num, -1)

        y     = self.Classifier(self.Temporal_Block(x_vec).reshape(x.shape[0], -1))

        return y, x_csp, x_log


class Graph_CSPNet_Basic(nn.Module):

    def __init__(self, channel_num, P, mlp, dataset = 'KU'):
        super(Graph_CSPNet_Basic, self).__init__()

        self._mlp       = mlp
        self.channel_in = channel_num
        self.P          = P

        if dataset   == 'KU':
            classes     = 2
            self.dims   = [20, 30, 30, 20]
        elif dataset == 'BNCI2014001':
            classes     = 4
            self.dims   = [22, 36, 36, 22]
        elif dataset == 'BNCI2015001':
            classes     = 2
            self.dims   = [13, 30, 30, 13]

        self.Graph_BiMap_Block = self._make_graph_bimap_block(len(self.dims)//2)
        self.LogEig            = LogEig()
        
        if self._mlp:
            self.Classifier = nn.Sequential(
            nn.Linear(channel_num*dims[-1]**2, channel_num),
            nn.ReLU(inplace=True),
            nn.Linear(channel_num, channel_num),
            nn.ReLU(inplace=True),
            nn.Linear(channel_num, classes)
            ).double()
        else:
            self.Classifier = nn.Linear(channel_num*self.dims[-1]**2, classes).double()

    def _make_graph_bimap_block(self, layer_num):

        layers = []
        _I     = th.eye(self.P.shape[0], dtype=th.double)
        
        dim_in, dim_out = self.dims[0], self.dims[1]
        layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, self.P))
        layers.append(ReEig())

        for i in range(1, layer_num):
            dim_in, dim_out = self.dims[2*i], self.dims[2*i+1]
            layers.append(Graph_BiMap(self.channel_in, dim_in, dim_out, _I))
            layers.append(ReEig())

        return nn.Sequential(*layers).double()


    def forward(self, x):

        x_csp = self.Graph_BiMap_Block(x)

        x_log = self.LogEig(x_csp)

        # NCHW Format (batch_size, window_num*band_num, 4, 4) ---> (batch_size, 1, window_num, band_num * 4 * 4)
        x_vec = x_log.view(x_log.shape[0], -1)

        y     = self.Classifier(x_vec)

        return y, x_csp, x_log

