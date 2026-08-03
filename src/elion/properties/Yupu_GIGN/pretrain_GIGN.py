
# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from predataset_GIGN import PreGraphDataset,PreGraphDatasetV
from torch_geometric.loader import DataListLoader, DataLoader
# from torch_geometric.loader import Data
from torch_geometric.nn import DataParallel, global_max_pool,global_mean_pool, GINConv
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
import time
from infoloss import InfoNCE, InfoNCESep
print("Available GPUs:", torch.cuda.device_count())

# %%
def val(model,model1, dataloader, device):
    model.eval()
    criterion = InfoNCE(temperature=0.1,batch_size = 256,n_views=11)
    criterion1 = InfoNCESep(temperature=0.1,batch_size = 256,n_views=11)
    pred_list = []
    label_list = []
    for data in dataloader:
        # data = data.to(device)
        # start = time.time()
        with torch.no_grad():
            InputBatch1= []
            InputBatch2= []
            for b in data:
                    # if len(b)==5:
                    #     InputBatch1 +=b
                    #     # InputBatch2 +=[b[0]]+b[11:]
                    # else:
                    #     continue
                InputBatch1 +=b
                # InputBatch2 +=[b[0]]+b[11:]
            InputBatch1 = [data.cuda() for data in InputBatch1]
            # InputBatch2 = [data.cuda() for data in InputBatch2]
            # print()
            # data = data.to(device)
            # print([data1[0].y for data1 in data])
            pred,label,x1 = model(InputBatch1)
            pred1,label1,x11 = model1(InputBatch1)
            # label = data.y
            x1 = x1.view(-1,11,256)
            x11 = x11.view(-1,11,256)
            loss = criterion(x1, label)
            loss1 = criterion1(x1,x11,label)
            loss =0.1*loss+loss1
            label_list.append(loss)
            # pred_list.append(pred.detach().cpu().numpy())
            # label_list.append(label.detach().cpu().numpy())
        # end = time.time()
        # print(f"val:{start-end}")
    # pred = np.concatenate(pred_list, axis=0)
    # label = np.concatenate(label_list, axis=0)

    # coff = np.corrcoef(pred, label)[0, 1]
    # rmse = np.sqrt(mean_squared_error(label, pred))
    rmse = sum(label_list)/len(label_list)
    model.train()

    return rmse, 1

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN'
    
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    # batch_size = args.get("batch_size")
    batch_size = 256
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    epochs = 20
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    repeat = 5
    train_list = []
    val_list = []
    for repeat in range(repeats):
        args['repeat'] = repeat

        train_dir = os.path.join(data_root, 'train')
        valid_dir = os.path.join(data_root, 'valid')
        test2013_dir = os.path.join(data_root, 'test2013')
        test2016_dir = os.path.join(data_root, 'test2016')
        test2019_dir = os.path.join(data_root, 'test2019')

        train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
        test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
        test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
        test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))

        # train_set = PreGraphDataset(train_dir, train_df, graph_type=graph_type, create=False)
        
        # valid_set = PreGraphDatasetV(valid_dir, valid_df, graph_type=graph_type, create=False)
        train_set = PreGraphDataset("/blue/zhe.jiang/y.zhang1/proteinData/pretrain521", train_df, graph_type=graph_type, create=False)
        valid_set = PreGraphDataset("/blue/zhe.jiang/y.zhang1/proteinData/pretrain521", valid_df, graph_type=graph_type, create=False)
        
        train_set.data_name = train_set.data_name[:-1000]
        valid_set.data_name = train_set.data_name[-1000:]
        train_set.load()
        valid_set.load()
        test2013_set = PreGraphDatasetV(test2013_dir, test2013_df, graph_type=graph_type, create=False)
        test2016_set = PreGraphDatasetV(test2016_dir, test2016_df, graph_type=graph_type, create=False)
        test2019_set = PreGraphDatasetV(test2019_dir, test2019_df, graph_type=graph_type, create=False)
        
        # train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True, num_workers=32,pin_memory = True)
        
        train_loader = DataListLoader(train_set,batch_size = batch_size,shuffle = True)
        
        valid_loader = DataListLoader(valid_set, batch_size=batch_size, shuffle=False)
        test2016_loader = DataListLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True,timeout = 2000)
        test2013_loader = DataListLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True,timeout = 2000)
        test2019_loader = DataListLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True,timeout = 2000)

        logger = TrainLogger(args,'preinfonce', cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")
        logger.info(f"test2019 data: {len(test2019_set)}")
        with open(os.path.join(logger.get_model_dir(),f'repeats_{repeat}.pkl'), 'wb') as file1:
            pickle.dump(train_set,file1)
        # device = torch.device('cuda:0')
        model = GIGN(35, 256,3).cuda()
        # model.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240521_132928_GIGN_repeat3/model/epoch-128, train_loss-0.3497, train_rmse-0.5913, valid_rmse-1.1848, valid_pr-0.7743.pt"))
        model.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240531_160944_GIGN_repeat6_finetuning/model/epoch-204, train_loss-0.2044, train_rmse-0.4521, valid_rmse-1.2244, valid_pr-0.7698.pt"))

        model = DataParallel(model)
        # for name, param in model.named_parameters():
        #     if ('output' in name ):
        #         # param.requires_grad = False
        #         # print(param.shape)
        #         continue
        #     else:
        #         if len(param.shape)>1:
        #             nn.init.kaiming_normal_(param)
        model1 = GIGN(35, 256,3).cuda()
        # model1.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240521_132928_GIGN_repeat3/model/epoch-128, train_loss-0.3497, train_rmse-0.5913, valid_rmse-1.1848, valid_pr-0.7743.pt"))
        model1.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240531_160944_GIGN_repeat6_finetuning/model/epoch-204, train_loss-0.2044, train_rmse-0.4521, valid_rmse-1.2244, valid_pr-0.7698.pt"))
        model1 = DataParallel(model1)
        
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()
        criterion = InfoNCE(temperature=0.1,batch_size = batch_size,n_views=1)
        criterion1 = InfoNCESep(temperature=0.1,batch_size = batch_size,n_views=1)
        
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        
        model.train()
        for epoch in range(epochs):
            # start = time.time()
            for data in train_loader:
                
                InputBatch1 = []
                # InputBatch2 = []
                for b in data:
                    # if b[0]!=torch.ones(1,1):
                    #     InputBatch1 +=b
                    #     # InputBatch2 +=[b[0]]+b[11:]
                    # else:
                    #     continue
                    #     # InputBatch1 +=b
                    #     # InputBatch2 +=b
                    InputBatch1+=b
                # print(len(InputBatch1))
                InputBatch1 = [data.cuda() for data in InputBatch1]
                # InputBatch2 = [data.cuda() for data in InputBatch2]
                # data = data.to(device)
                # print([data1[0].y for data1 in data])
                # print(len(InputBatch1))
                pred,label,x1 = model(InputBatch1)
                pred1,label1,x11 = model1(InputBatch1)
                # # # label = data.y
                x1 = x1.view(-1,1,256)
                x11 = x11.view(-1,1,256)
                loss = criterion(x1, label)
                loss1 = criterion1(x1,x11,label)
                if epoch<10:
                    loss =0.01*loss+loss1
                else:
                    loss = loss+0.5*loss1
                # break
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                running_loss.update(loss.item(), label.size(0)) 
                # print(len(InputBatch1))
                
                
            # break
            # end = time.time()
            # print(f"epoch:{end-start}")
            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()
            # start validating
            valid_rmse, valid_pr = val(model,model1, valid_loader, 'cuda0')
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            train_list.append(epoch_rmse.item())
            val_list.append(valid_rmse.item())
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        # final testing
        load_model_dict(model, best_model_list[-1])
        device = 'cuda'
        valid_rmse, valid_pr = val(model,model1, valid_loader, device)
        # test2013_rmse, test2013_pr = val(model, test2013_loader, device)
        # test2016_rmse, test2016_pr = val(model, test2016_loader, device)
        # test2019_rmse, test2019_pr = val(model, test2019_loader, device)
        save_path = os.path.join(logger.get_model_dir(), "loss.pdf")
        plot_training_curve(line_1_x=range(len(train_list)),
						line_1_y=train_list,
						line_2_x=range(len(val_list)),
						line_2_y=val_list,
						save_path=save_path,
						y_label="RMSE")
        # msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f," \
        #             % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)

        # logger.info(msg)

# %%
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from utils import AverageMeter
# from GIGN import GIGN
# from predataset_GIGN import PreGraphDataset,PreGraphDatasetV
# from torch_geometric.loader import DataListLoader
# from torch_geometric.nn import DataParallel, global_max_pool,global_mean_pool, GINConv
# from config.config_dict import Config
# from log.train_logger import TrainLogger
# import numpy as np
# import random
# from utils import *
# from sklearn.metrics import mean_squared_error
# from infoloss import InfoNCE, InfoNCESep

# # %%
# def val(model, dataloader, device):
#     model.eval()

#     pred_list = []
#     label_list = []
#     for data in dataloader:
#         data = data.to(device)
#         with torch.no_grad():
#             pred,y,x1 = model(data)
#             label = data.y

#             pred_list.append(pred.detach().cpu().numpy())
#             label_list.append(label.detach().cpu().numpy())
            
#     pred = np.concatenate(pred_list, axis=0)
#     label = np.concatenate(label_list, axis=0)

#     coff = np.corrcoef(pred, label)[0, 1]
#     rmse = np.sqrt(mean_squared_error(label, pred))

#     model.train()

#     return rmse, coff

# # %%
# if __name__ == '__main__':
#     cfg = 'TrainConfig_GIGN'
#     config = Config(cfg)
#     args = config.get_config()
#     graph_type = args.get("graph_type")
#     save_model = args.get("save_model")
#     batch_size = args.get("batch_size")
#     data_root = args.get('data_root')
#     epochs = args.get('epochs')
#     epochs = 300
#     repeats = args.get('repeat')
#     early_stop_epoch = args.get("early_stop_epoch")
#     repeats = 10
#     test2013_list = []
#     test2016_list = []
#     test2019_list = []
#     test2013_p = []
#     test2016_p = []
#     test2019_p = []
    
#     for repeat in range(repeats):
#         args['repeat'] = repeat
#         train_list = []
#         val_list = []
#         train_dir = os.path.join(data_root, 'train')
#         valid_dir = os.path.join(data_root, 'valid')
#         test2013_dir = os.path.join(data_root, 'test2013')
#         test2016_dir = os.path.join(data_root, 'test2016')
#         test2019_dir = os.path.join(data_root, 'test2019')

#         train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
#         valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
#         test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
#         test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
#         test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))

#         train_set = PreGraphDataset(train_dir, train_df, graph_type=graph_type, create=False)
#         valid_set = PreGraphDatasetV(valid_dir, valid_df, graph_type=graph_type, create=False)
#         train_set = PreGraphDataset("/blue/zhe.jiang/y.zhang1/proteinData/pretrain521", train_df, graph_type=graph_type, create=False)
#         valid_set = PreGraphDataset("/blue/zhe.jiang/y.zhang1/proteinData/pretrain521", valid_df, graph_type=graph_type, create=False)
#         train_set.data_name = train_set.data_name[:-1000]
#         valid_set.data_name = train_set.data_name[-1000:]
#         test2013_set = PreGraphDatasetV(test2013_dir, test2013_df, graph_type=graph_type, create=False)
#         test2016_set = PreGraphDatasetV(test2016_dir, test2016_df, graph_type=graph_type, create=False)
#         test2019_set = PreGraphDatasetV(test2019_dir, test2019_df, graph_type=graph_type, create=False)
#         all_path = train_set.graph_paths+valid_set.graph_paths
#         random.shuffle(all_path)
#         print(len(all_path))
        
#         train_loader = DataListLoader(train_set,batch_size = batch_size,shuffle = True,num_workers=24,pin_memory = True)
#         valid_loader = DataListLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
#         test2016_loader = DataListLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#         test2013_loader = DataListLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#         test2019_loader = DataListLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#         logger = TrainLogger(args,'preinfonce', cfg, create=True)
#         logger.info(__file__)
#         logger.info(f"train data: {len(train_set)}")
#         logger.info(f"valid data: {len(valid_set)}")
#         logger.info(f"test2013 data: {len(test2013_set)}")
#         logger.info(f"test2016 data: {len(test2016_set)}")
#         logger.info(f"test2019 data: {len(test2019_set)}")
#         # with open(os.path.join(logger.get_model_dir(),f'repeats_{repeat}.pkl'), 'wb') as file1:
#         #     pickle.dump(all_path,file1)
#         # with open(os.path.join(f'/blue/zhe.jiang/y.zhang1/GIGN/model/',f'repeats_{repeat}.pkl'), 'rb') as file1:
#         #     all_path = pickle.load(file1)
#         device = torch.device('cuda:0')
#         model = GIGN(35, 256,3).cuda()
#         # model.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240521_132928_GIGN_repeat3/model/epoch-128, train_loss-0.3497, train_rmse-0.5913, valid_rmse-1.1848, valid_pr-0.7743.pt"))
#         model.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240531_112358_GIGN_repeat0_finetuning/model/epoch-241, train_loss-0.1736, train_rmse-0.4167, valid_rmse-1.2254, valid_pr-0.7604.pt"))

#         model = DataParallel(model)
# #         for name, param in model.named_parameters():
# #             if ('output' in name ):
# #                 # param.requires_grad = False
# #                 # print(param.shape)
# #                 continue
# #             else:
# #                 if len(param.shape)>1:
# #                     nn.init.kaiming_normal_(param)
#         model1 = GIGN(35, 256,3).cuda()
#         # model1.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240521_132928_GIGN_repeat3/model/epoch-128, train_loss-0.3497, train_rmse-0.5913, valid_rmse-1.1848, valid_pr-0.7743.pt"))
#         model1.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240531_112358_GIGN_repeat0_finetuning/model/epoch-241, train_loss-0.1736, train_rmse-0.4167, valid_rmse-1.2254, valid_pr-0.7604.pt"))
#         model1 = DataParallel(model1)
#         # for name, param in model.named_parameters():
#         #     if ('output' in name and 'weight' in name):
#         #         # param.requires_grad = False
#         #         # print(param.shape)
#         #         if len(param.shape)>1:
#         #             nn.init.kaiming_normal_(param)
#         #         print(name)
#         #     else:
#         #         if len(param.shape)>1:
#         #             param.requires_grad = False
#         optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience =40,factor=0.1,min_lr = 0.00001)
#         criterion = nn.MSELoss()
#         criterion = InfoNCE(temperature=0.1,batch_size = 256,n_views=11)
#         criterion1 = InfoNCESep(temperature=0.1,batch_size = 256,n_views=11)
#         running_loss = AverageMeter()
#         running_acc = AverageMeter()
#         running_best_mse = BestMeter("min")
#         best_model_list = []
        
#         model.train()
#         for epoch in range(epochs):
#             for data in train_loader:
#                 InputBatch1 = []
#                 # InputBatch2 = []
#                 for b in data:
#                     if len(b)!=0:
#                         InputBatch1 +=b
#                         # InputBatch2 +=[b[0]]+b[11:]
#                     else:
#                         InputBatch1 +=b
#                         # InputBatch2 +=b
                # InputBatch1 = [data.cuda() for data in InputBatch1]
#                 # InputBatch2 = [data.cuda() for data in InputBatch2]
#                 # print()
#                 # data = data.to(device)
#                 # print([data1[0].y for data1 in data])
#                 pred,label,x1 = model(InputBatch1)
#                 pred1,label1,x11 = model1(InputBatch1)
#                 # label = data.y
#                 x1 = x1.view(-1,11,256)
#                 x11 = x11.view(-1,11,256)
#                 loss = criterion(x1, label)
#                 loss1 = criterion1(x1,x11,label)
#                 loss =0.1*loss+loss1
#                 # break
#                 optimizer.zero_grad()
#                 loss.backward()

#                 optimizer.step()
#                 running_loss.update(loss.item(), label.size(0)) 

#             epoch_loss = running_loss.get_average()
#             epoch_rmse = np.sqrt(epoch_loss)
#             running_loss.reset()
            
#             # start validating
#             valid_rmse,valid_pr = [1,1]
#             # valid_rmse, valid_pr = val(model, valid_loader, device)
#             msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
#                     % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
#             train_list.append(epoch_rmse)
#             val_list.append(valid_rmse)
#             logger.info(msg)
#             scheduler.step(valid_rmse)
#             if valid_rmse < running_best_mse.get_best():
#                 running_best_mse.update(valid_rmse)
#                 if save_model:
#                     msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
#                     % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
#                     model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
#                     best_model_list.append(model_path)
#                     save_model_dict(model, logger.get_model_dir(), msg)
#             else:
#                 count = running_best_mse.counter()
#                 if count > early_stop_epoch:
#                     best_mse = running_best_mse.get_best()
#                     msg = "best_rmse: %.4f" % best_mse
#                     logger.info(f"early stop in epoch {epoch}")
#                     logger.info(msg)
#                     break_flag = True
#                     break

#         # final testing
#         load_model_dict(model, best_model_list[-1])
#         valid_rmse, valid_pr = val(model, valid_loader, device)
#         test2013_rmse, test2013_pr = val(model, test2013_loader, device)
#         test2016_rmse, test2016_pr = val(model, test2016_loader, device)
#         test2019_rmse, test2019_pr = val(model, test2019_loader, device)
#         test2013_list.append(test2013_rmse)
#         test2016_list.append(test2016_rmse)
#         test2019_list.append(test2019_rmse)
#         test2013_p.append(test2013_pr)
#         test2016_p.append(test2016_pr)
#         test2019_p.append(test2019_pr)
#         save_path = os.path.join(logger.get_model_dir(), "loss.pdf")
#         plot_training_curve(line_1_x=range(len(train_list)),
# 						line_1_y=train_list,
# 						line_2_x=range(len(val_list)),
# 						line_2_y=val_list,
# 						save_path=save_path,
# 						y_label="RMSE")
#         msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f," \
#                     % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)
#         print(test2013_list)
#         print(test2016_list)
#         print(test2019_list)
#         print(test2013_p)
#         print(test2016_p)
#         print(test2019_p)
#         logger.info(msg)
        

# %%
        