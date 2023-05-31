'''
#####################################################################################################################
Author      : Ce Ju
Date        : 1st, Jun., 2023
---------------------------------------------------------------------------------------------------------------------
Descriptions:  



This code originates from modifications of the work presented in this paper and the SPDSW code, mainly implementing the 
EMD method. Additionally, we have specifically used the KU dataset as an example for adaptation.



#######################################################################################################################
'''



import torch as th
import geoopt 
import ot

from pyriemann.utils.mean import mean_riemann


def EMD_SPD(Xs, Xt, metric="le"):
    """
        Xs: Source (n_batch, d, d)
        Xt: Target (m_batch, d, d)
        metric: "ai" or "le"
    """
    d = Xs.shape[-1]
    device = Xs.device 
    
    if metric == "ai":
        manifold_spd = geoopt.SymmetricPositiveDefinite("AIM")
    elif metric == "le":
        manifold_spd = geoopt.SymmetricPositiveDefinite("LEM")        

    a = th.ones((len(Xs),), device=device, dtype=th.float64)/len(Xs)
    b = th.ones((len(Xt),), device=device, dtype=th.float64)/len(Xt)
    M = manifold_spd.dist(Xs[:,None], Xt[None])**2
    
    P = ot.emd(a, b, M)
    
    if metric == "ai":
        cpt = th.zeros((len(Xs), d, d))
        for i in range(len(Xs)):
            cpt[i] = th.tensor(mean_riemann(Xt.cpu().numpy(), sample_weight=P[i].cpu().numpy()))
    elif metric == "le":
        log_Xt = geoopt.linalg.sym_logm(Xt)            
        cpt    = th.matmul(P, log_Xt[None].reshape(-1, d*d)).reshape(-1,d,d)
        cpt    = geoopt.linalg.sym_expm(cpt*len(Xt))
            
    return cpt

if __name__ == '__main__':

    channel_index = [7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    x_train = np.load('Dataset/sess1_sub1_x.npy')[:   , channel_index, :]
    x_val   = np.load('Dataset/sess2_sub1_x.npy')[:100, channel_index, :]
    x_test  = np.load('Dataset/sess2_sub1_x.npy')[100:, channel_index, :]
    
    fbank   = FilterBank(fs=1000)
    _       = fbank.get_filter_coeff()

    x_train = fbank.filter_data(x_train, window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
    x_val   = fbank.filter_data(x_val,   window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
    x_test  = fbank.filter_data(x_test,  window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
    
    def temporal_sig2cov(x):

        temp_stack  = []
        for i in range(x.shape[0]):
            cov_stack = []
            for j in range(x.shape[1]):
              cov_stack.append(Shrinkage(1e-2).transform(Covariances().transform(x[i,j])))
            temp_stack.append(np.stack(cov_stack, axis = 0))

        return np.stack(temp_stack, axis = 0)
    

    def stack(data, interval):

        data_record = []
        for [a, b] in interval:
            data_record.append(np.expand_dims(data[:, :, :, a:b], axis = 1))

        return temporal_sig2cov(np.concatenate(data_record, axis = 1))

    interval = [[0, 2500]]  
    
    train = th.from_numpy(stack(x_train, interval).reshape(200, 9, 20, 20))
    val   = th.from_numpy(stack(x_val, interval).reshape(100, 9, 20, 20))
    test  = th.from_numpy(stack(x_test, interval).reshape(100, 9, 20, 20))

    train_ot = th.zeros(200, 9, 20, 20)
    val_ot   = th.zeros(100, 9, 20, 20)

    for channel in range(train_ot.shape[1]):
        train_ot[:, channel, :, :] = EMD_SPD(train[:, channel, :, :], test[:, channel, :, :], metric="le")
        val_ot[:, channel, :, :]   = EMD_SPD(val[:, channel, :, :], test[:, channel, :, :], metric="le")





    
