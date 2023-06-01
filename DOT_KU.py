'''
#####################################################################################################################
Date       : 1st, Sep., 2022
Discription: Trainning file of DOT for the holdout scenario on the Korea University dataset. 
#######################################################################################################################
'''
import time
import pandas as pd
import numpy as np
import argparse


from torch.autograd import Variable
import torch.nn.functional as F
import torch as th
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from sklearn.model_selection import StratifiedShuffleSplit



from utils.DOT import DOT_LEM
from utils.early_stopping import EarlyStopping
from utils.load_data import load_KU, dataloader_in_main
import utils.geoopt as geoopt



def adjust_learning_rate(optimizer, epoch):
    optimizer.lr = args.initial_lr * (args.decay ** (epoch // 100))


def main(args, train, val, test, train_y, val_y, test_y, pseudo_test_y, sub, total_sub):

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device   = torch.device("cuda" if use_cuda else "cpu")


    train       = Variable(torch.from_numpy(train)).double()
    val         = Variable(torch.from_numpy(val)).double()
    test        = Variable(torch.from_numpy(test)).double()

    train_y     = Variable(torch.LongTensor(train_y))
    val_y       = Variable(torch.LongTensor(val_y))
    test_y      = Variable(torch.LongTensor(test_y))
    pseudo_test_y = Variable(torch.LongTensor([*val_y, *pseudo_test_y]))
      
    train_dataset = dataloader_in_main(train, train_y)
    val_dataset   = dataloader_in_main(val,  val_y)
    test_dataset  = dataloader_in_main(test, test_y)
    pseudo_test_dataset = My_Dataset(torch.cat((val, test), 0), pseudo_test_y)



    train_kwargs = {'batch_size': args.train_batch_size}
    if use_cuda:
          cuda_kwargs ={'num_workers': 100,
                        'sampler': train_sampler,
                          'pin_memory': True,
                          'shuffle': True     
          }
          train_kwargs.update(cuda_kwargs)
          
    valid_kwargs = {'batch_size': args.valid_batch_size}
    if use_cuda:
          cuda_kwargs ={'num_workers': 100,
                        'sampler':valid_sampler,
                          'pin_memory': True,
                          'shuffle': True     
          }
          valid_kwargs.update(cuda_kwargs)

    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
          cuda_kwargs ={'num_workers': 100,
                          'pin_memory': True,
                          'shuffle': True      
          }
          test_kwargs.update(cuda_kwargs)

    pseudo_test_kwargs = {'batch_size': args.test_batch_size_for_adaption}
        if use_cuda:
            cuda_kwargs ={'num_workers': 100,
                           'pin_memory': True,
                           'shuffle': True      
            }
            pseudo_test_kwargs.update(cuda_kwargs)

    train_loader       = torch.utils.data.DataLoader(dataset= train_dataset, **train_kwargs)
    valid_loader       = torch.utils.data.DataLoader(dataset= train_dataset, **valid_kwargs)
    test_loader        = torch.utils.data.DataLoader(dataset= test_dataset,  **test_kwargs)
    pseudo_test_loader = torch.utils.data.DataLoader(dataset= pseudo_test_dataset,  **pseudo_test_kwargs)



    model = DOT_LEM(channel_num = train.shape[1]*train.shape[2], 
        DOT_type = 'MDA',
        architecture = 'Tensor',
        dataset = 'KU',
        ).to(device)

    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.initial_lr)


    early_stopping = EarlyStopping(
        alg_name = args.alg_name, 
        path_w   = args.weights_folder_path + args.alg_name + '_checkpoint.pt', 
        patience = args.patience, 
        verbose  = True, 
        )


    print('#####Start Trainning######')

    for epoch in range(1, args.epochs+1):

        adjust_learning_rate(optimizer, epoch)

        model.train()

        train_correct = 0

        pseudo_tgt_iter = iter(pseudo_test_loader)
    
        for batch_idx, (batch_train, batch_train_y) in enumerate(train_loader):

            optimizer.zero_grad()

            try:
              pseudo_tgt_data, pseudo_tgt_label  = pseudo_tgt_iter.next()
            except Exception as err:
              pseudo_tgt_iter                    = iter(pseudo_test_loader)
              pseudo_tgt_data, pseudo_tgt_label  = next(pseudo_tgt_iter)

            logits = model(batch_train.to(device), batch_train_y.to(device), pseudo_tgt_data.to(device), pseudo_tgt_label.to(device))
            output = F.log_softmax(logits, dim = -1)
            loss   = F.nll_loss(output, batch_train_y.to(device))

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('----#------#-----#-----#-----#-----#-----#-----')
                pred    = output.data.max(1, keepdim=True)[1]
                train_correct += pred.eq(batch_train_y.to(device).data.view_as(pred)).long().cpu().sum()
                torch.save(model.state_dict(), args.weights_folder_path + args.alg_name+'_model.pth')
                torch.save(optimizer.state_dict(), args.weights_folder_path+'optimizer.pth')


                print('['+args.alg_name+': Sub No.{}/{}, Epoch {}/{}, Completed {:.0f}%]:\nTrainning loss {:.10f} Acc.: {:.4f}'.format(\
                        sub, total_sub, epoch, args.epochs, 100. * (1+batch_idx) / len(train_loader), loss.cpu().detach().numpy(),\
                        train_correct.item()/len(train_loader.dataset)))
                    

        #Validate the Model
        valid_losses  = []
        valid_loss    =  0
        valid_correct =  0

        model.eval()

        for batch_idx, (batch_valid, batch_valid_y) in enumerate(valid_loader):


            try:
                pseudo_tgt_data, pseudo_tgt_label  = pseudo_tgt_iter.next()
            except Exception as err:
                pseudo_tgt_iter                    = iter(pseudo_test_loader)
                pseudo_tgt_data, pseudo_tgt_label  = next(pseudo_tgt_iter)


            logits         = model(batch_valid.to(device), batch_valid_y.to(device), pseudo_tgt_data.to(device), pseudo_tgt_label.to(device))
            output         = F.log_softmax(logits, dim = -1)
            valid_loss    += F.nll_loss(output, batch_valid_y.to(device))
            valid_losses.append(valid_loss.item())
            
            
            pred           = output.data.max(1, keepdim=True)[1]
            valid_correct += pred.eq(batch_valid_y.to(device).data.view_as(pred)).long().cpu().sum()

        print('Validate loss: {:.10f} Acc: {:.4f}'.format(sum(valid_losses), valid_correct.item()/len(valid_loader.dataset)))
        
        early_stopping(np.average(valid_losses), model)
        
        if early_stopping.early_stop:
          print("Early Stopping!")
          break

        

    #Testing
    print('###############################################################')
    print('START TESTING')
    print('###############################################################')

    
    model.eval()

    test_loss    = 0
    test_correct = 0

    with torch.no_grad():
        for batch_idx, (batch_test, batch_test_y) in enumerate(test_loader):

            logits        = model(batch_test.to(device), batch_test_y.to(device), batch_test.to(device), batch_test_y.to(device))
            output        = F.log_softmax(logits, dim = -1)
            test_loss    += F.nll_loss(output, batch_test_y.to(device))
            
            test_pred     = output.data.max(1, keepdim=True)[1]
            test_correct += test_pred.eq(batch_test_y.to(device).data.view_as(test_pred)).long().cpu().sum()

            print('-----------------------------------')
            print('Testing Batch {}:'.format(batch_idx))
            print('  Pred Label:', test_pred.view(1, test_pred.shape[0]).cpu().numpy()[0])
            print('Ground Truth:', batch_test_y.numpy())


    return test_correct.item()/len(test_loader.dataset), test_loss.item()/len(test_loader.dataset)

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg_name', default = 'Tensor_CSPNet', help = 'name of model')
    
    parser.add_argument('--no-cuda',  action = 'store_true', default=False, help = 'disables CUDA training')
    parser.add_argument('--initial_lr', type = float, default = 1e-3,       help = "initial_lr")
    parser.add_argument('--decay',      type = float, default = 1, help= "decay rate for adjust_learning")

    parser.add_argument('--start_No', type=int, default = 1,  help = 'testing starts on subject #')
    parser.add_argument('--end_No',   type=int, default = 1,  help = 'testing ends on subject #')
    parser.add_argument('--epochs',   type=int, default = 50, help = 'number of epochs to train')
    parser.add_argument('--patience', type=int, default = 10, help = 'patience for early stopping')

    parser.add_argument('--train_batch_size', type = int, default = 72, help = 'batch size in each epoch for trainning')
    parser.add_argument('--test_batch_size',  type = int, default = 72, help = 'batch size in each epoch for testing')
    parser.add_argument('--valid_batch_size', type = int, default = 72, help = 'batch size in each epoch for validation')
    parser.add_argument('--test_batch_size_for_adaption', type=int,  default = 200, help='batch size in each epoch for Trainning')

    parser.add_argument('--seed',         type = int, default = 1, metavar='S', help = 'random seed (default: 1)')
    parser.add_argument('--log_interval', type = int, default = 1, help = 'how many batches to wait before logging training status')
    parser.add_argument('--save-model', action = 'store_true', default=False, help = 'for saving the current model')

    parser.add_argument('--folder_name',         default = 'results')
    parser.add_argument('--weights_folder_path', default = 'model_paras/')

    args = parser.parse_args(args=[])
    return args


if __name__ == '__main__':

    args   = args_parser()

    alg_df = pd.DataFrame(columns=['Test Acc'])

    print('############Start Task#################')
    
    for sub in range(args.start_No, args.end_No + 1):

        KU_dataset   = load_KU(['dataset/KU_sess1_sub'+str(sub), 'dataset/KU_sess2_sub'+str(sub)],
            '', 
            alg_name = args.alg_name,
            scenario = 'Holdout'
            )

        alg_record   = []

        start        = time.time()

        x_train_stack, x_val_stack, x_test_stack, y_train, y_val, y_test = KU_dataset.generate_training_valid_test_set_Holdout()
        
        #Pseudo label predicted by Graph-CSPNet.
        pseudo_y_test = [int(char) for char in pd.read_csv('Graph_KU_S2S_pred_y.csv')['Pred_y'][sub-1][1:-1].split()]


        acc, loss = main(
            args       = args, 
            train      = x_train_stack, 
            val        = x_val_stack,
            test       = x_test_stack, 
            train_y    = y_train,
            val_y      = y_val,  
            test_y     = y_test,
            pseudo_test_y = pseudo_y_test, 
            sub        = sub, 
            total_sub  = args.end_No - args.start_No + 1
            )

        print('##############################################################')

        print(args.alg_name + ' Testing Loss.: {:4f} Acc: {:4f}'.format(loss, acc))

        end = time.time()

        alg_record.append(acc)
        
        alg_df.loc[sub] = alg_record
 
        alg_df.to_csv(args.folder_name + '/' \
        + time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()) \
        + args.alg_name \
        +'_Sub(' \
        + str(args.start_No) \
        +'-' \
        +str(args.end_No) \
        +')' \
        +'_' \
        + str(args.epochs)\
        + '.csv'\
        , index = False)
    

