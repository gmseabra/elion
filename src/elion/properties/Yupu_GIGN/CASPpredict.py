
# # %%
import csv
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import AverageMeter
from GIGN import GIGN
from predataset_GIGN import PreGraphDatasetD,PreGraphDatasetV
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel, global_max_pool,global_mean_pool, GINConv
from config.config_dict import Config
from log.train_logger import TrainLogger
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from infoloss import InfoNCE,InfoNCESep
print("Available GPUs:", torch.cuda.device_count())

# %%
def val(model,model1, dataloader, device):
    model.eval()
    criterion2 = nn.MSELoss()
    criterion = InfoNCE(temperature=0.1,batch_size = 16,n_views=11)
    criterion1 = InfoNCESep(temperature=0.1,batch_size = 16,n_views=11)
    pred_list = []
    label_list = []
    for data in dataloader:
        # data = data.to(device)
        with torch.no_grad():
            InputBatch1 = []
            InputBatch2 = []
            for b in data:
                if len(b)!=0:
                    InputBatch1 +=b[:11]
                    InputBatch2 +=b[11:]
                else:
                    InputBatch1 +=b
                    InputBatch2 +=b
            InputBatch1 = [data.cuda() for data in InputBatch1]
            InputBatch2 = [data.cuda() for data in InputBatch2]
            for item in InputBatch2:
                item.pos.requires_grad_()
            pred,label,x1 = model(InputBatch1)
            pred1,label1,x11 = model1(InputBatch1)
            # label = data.y
            x1 = x1.view(-1,11,256)
            x11 = x11.view(-1,11,256)
            loss = criterion(x1, label)
            loss1 = criterion1(x1,x11,label)
            loss =0.5*loss+loss1
            pred2,label2,x12 = model(InputBatch2)
            # loss2 = 0.5*denoising_loss(pred2,InputBatch2,label2,2,batch_size,criterion2)
            # loss += loss2
            label_list.append(loss)
            # pred_list.append(pred.detach().cpu().numpy())
            # label_list.append(label.detach().cpu().numpy())
            
    # pred = np.concatenate(pred_list, axis=0)
    # label = np.concatenate(label_list, axis=0)

    # coff = np.corrcoef(pred, label)[0, 1]
    # rmse = np.sqrt(mean_squared_error(label, pred))
    rmse = sum(label_list)/len(label_list)
    model.train()

    return rmse, 1
def denoising_loss(y,data,u,std,batch_size,cri):
    loss = []
    # print(len(data))
    for i in range(len(data)//10):
        for j in range(10):
            d = data[i*10+j].pos
            # pred,label,x1 = model(data[i*batch_size+j])
            f = torch.autograd.grad(y[i*10+j], d, create_graph=True, retain_graph=True)[0]
            u1 = u[i*10+j]
            
            target = (u1/std)*torch.ones_like(d).cuda()
            loss.append(cri(f,target).view(-1))
    # print(torch.cat(loss))
    return torch.cat(loss).mean()
                
import psutil

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN'
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    # random.seed(2024)
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    # batch_size = args.get("batch_size")
    batch_size = 2
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    epochs = 20
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")
    repeats = 1
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

        # train_set = PreGraphDatasetV(train_dir, train_df, graph_type=graph_type, create=False)
        # valid_set = PreGraphDatasetV(valid_dir, valid_df, graph_type=graph_type, create=False)
        train_set = PreGraphDatasetD("/blue/zhe.jiang/y.zhang1/CASP/data", train_df, graph_type=graph_type, create=False)
        valid_set = PreGraphDatasetD("/blue/zhe.jiang/y.zhang1/CASP/data", valid_df, graph_type=graph_type, create=False)
        
        valid_set.data_name = train_set.data_name[:]
        train_set.data_name = train_set.data_name[:]
        train_set.load()
        valid_set.load()
       
        
        
        test2013_set = PreGraphDatasetV(test2013_dir, test2013_df, graph_type=graph_type, create=False)
        test2016_set = PreGraphDatasetV(test2016_dir, test2016_df, graph_type=graph_type, create=False)
        test2019_set = PreGraphDatasetV(test2019_dir, test2019_df, graph_type=graph_type, create=False)

        train_loader = DataListLoader(train_set,batch_size = batch_size,shuffle = False)
        valid_loader = DataListLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test2016_loader = DataListLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test2013_loader = DataListLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test2019_loader = DataListLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        logger = TrainLogger(args,'pretrain_de', cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")
        logger.info(f"test2019 data: {len(test2019_set)}")
        with open(os.path.join(logger.get_model_dir(),f'repeats_{repeat}.pkl'), 'wb') as file1:
            pickle.dump([train_set.data_name,valid_set.data_name],file1)
        # device = torch.device('cuda:0')
        model = GIGN(35, 256,3).cuda()
        # model.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240521_132928_GIGN_repeat3/model/epoch-128, train_loss-0.3497, train_rmse-0.5913, valid_rmse-1.1848, valid_pr-0.7743.pt"))

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
        model1.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/GIGN/model/20240708_055218_GIGN_repeat2_finetuning/model/epoch-98, train_loss-0.4573, train_rmse-0.6762, valid_rmse-1.1939, valid_pr-0.8351.pt"))
        model1 = DataParallel(model1)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        criterion2 = nn.MSELoss()
        criterion = InfoNCE(temperature=0.1,batch_size = 16,n_views=11,alpha = 0.2*repeat)
        criterion1 = InfoNCESep(temperature=0.1,batch_size = 16,n_views=11)
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []
        device = 'cuda0'
        model.eval()
        print(len(train_set))
        print(len(train_set.datalist))
        save_pred = []
        save_name = []
        save_length = []
        save_max = []
        for epoch in range(epochs):
            
            for index, input_data in enumerate(train_loader):
                
                # data,name,length = input_data
                # print((input_data[1]))
                process = psutil.Process()
                InputBatch1 = []
                InputBatch = []
                for b in input_data:
                    if len(b)!=0:
                        InputBatch +=b[0]
                    else:
                        InputBatch +=b[0]
                # print(InputBatch[1])
                InputBatch1 = [data.cuda() for data in InputBatch]
                name = [data[1] for data in input_data]
                length = [data[2] for data in input_data]
                pred,label,x1 = model1(InputBatch1)
                pred = pred.to('cpu').detach().numpy().tolist()
                # print(pred)
                save_pred+=pred
                print(length)
                save_name +=name
                save_length += length
                # print(max(pred[:length[0]]))
                # print(max(pred[length[0]:length[0]+length[1]]))
                # print(length)
                # print(pred)
                save_max.append(max(pred[:length[0]]))
                try:
                    if (len(length)>1):
                        save_max.append(max(pred[length[0]:length[0]+length[1]]))
                except:
                    print(name)
                # print(len(length))
            final_data = []
            output_index = 0
            final_max = []
            for name, length in zip(save_name, save_length):
                for i in range(length):
                    final_data.append([f"{name}_{i+1}", save_pred[output_index]])
                    output_index += 1
            output_index = 0
            for name in save_name:
                final_max.append([f"{name}", save_max[output_index]])
                output_index += 1
            csv_filename = 'output.csv'
            with open(csv_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Name', 'Affinity(-log(kd/ki))'])
                csvwriter.writerows(final_data)
            csv_filename = 'max_value.csv'
            with open(csv_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Name', 'Affinity(-log(kd/ki))'])
                csvwriter.writerows(final_max)
            print(f"Data has been saved to {csv_filename}")
            break
        
        msg = 'finish'
        logger.info(msg)
        break
        
# %%import torch

# if __name__ == '__main__':
#     torch.cuda.manual_seed(1)
#     torch.cuda.manual_seed_all(1)
#     np.random.seed(1)
#     torch.manual_seed(1)
#     model = contrast(18,128,3,1)
#     now = datetime.now()
    
#     folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    
#     path = os.path.join("/blue/zhe.jiang/y.zhang1/saveModel/", folder_name+"_a2")
#     print(path)
    
#     os.makedirs(path, exist_ok=True)
#     # model.load_state_dict(torch.load('/blue/zhe.jiang/y.zhang1/saveModel/modelnewnewnew97.pth'))
#     # model.load_state_dict(torch.load("/blue/zhe.jiang/y.zhang1/saveModel1/modelnew97S.pth"))
#     # model =model.Net
#     model = model.cuda()
#     for name, param in model.named_parameters():
#         if name.startswith('layer1') or name.startswith('layer2'):
#             print(name)
#             param.requires_grad = False
#     model = DataParallel(model)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     # batch = 8
#     batch_size = 16
#     # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay = 0.000001, betas=(0.9, 0.95), eps=1e-8, amsgrad=False)
#     print(count_parameters(model))
#     data_file = []
#     # for i in os.listdir("/blue/zhe.jiang/y.zhang1/proteinData/trainContrast1"):
# 	   # if len(os.listdir(os.path.join("/blue/zhe.jiang/y.zhang1/proteinData/trainContrast1",i)))>30:
# 	   #     data_file.append(i)
#     # optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, momentum=0.0009)
#     criterion1 = torch.nn.MSELoss()
#     criterion = InfoNCE(temperature=0.001,batch_size = 64,n_views=10)
#     save_loss1 = []
#     save_loss2 = []
    
#     warm_up_iter = 20
#     T_max = 200	
#     lr_max = 0.001
#     lr_min = 1e-4	

#     lambda0 = lambda epoch: 1 if  epoch < warm_up_iter else \
#             0.1
    
#     # LambdaLR
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience = 8,factor=0.5)
#     time1 = []
    
#     # print(len(dataset.dataFile))
#     # for num in range(20):
#     dataset = generateDataset("/blue/zhe.jiang/y.zhang1/proteinData/trainContrast1_15/",size =60000,start=0)
#     dataLoader = DataListLoader(dataset,batch_size = 64,shuffle = True,num_workers=24,pin_memory = True)
#     datasetV = generateDataset("/blue/zhe.jiang/y.zhang1/proteinData/trainContrast1_15/",size=1000,start=60000)
#     dataLoaderV = DataListLoader(datasetV,batch_size = 20,shuffle = False,num_workers =2)
#     for epoch in range(30):
#         print(epoch)
#         save_1 = []
#         save_2 = []
#         # print(epoch)
#         # for batchs in dataset:
#         model.train()
#         start = time.time()
#         read_start = time.time()
#         read_end = time.time()
#         time_save=[0,0,0,0]
#         for batch_idx, batchs in enumerate(dataLoader):
#             read_end = time.time()
#             time_save[3]+=(read_end-read_start)
#             time_b = time.time()
#             time1 = time.time()    
#             InputBatch = []
            
#             for b in batchs:
#                 InputBatch +=b
#                 # print(len(b))
#             #
            
#             InputBatch = [data.cuda() for data in InputBatch]
#             time2 = time.time()
#             time_save[0]+=(time2-time1)
#             # with autocast():
#             time1 = time.time()
#             y,label,y1= model(InputBatch)
#             time2 = time.time()
#             time_save[1]+=(time2-time1)
#             # time1 = time.time()
            
#             # print("inference %f"%(time2-time1))
#             # print(y)
#             # y= model(batchs)
#             # print(torch.sum(x2[0:10,:]-x2[1:11,:]))
#             # print(batchs[1].x1[:1000,1:])
#             y = y.view(-1,10,128)
#             # y = y.permute(1,0,2)
#             # y = y.reshape(-1,128)
#             # y1 = y1.view(-1,1)
#             y1 = torch.squeeze(y1,1)
#             y1 = y1.view(-1,10,1)
            
#             # label = torch.unsqueeze(label,1)
#             # print(y)
#             # print(label)
#             # print("x1")
#             # print(x1)
#             # print("x2")
#             # print(x2)
#             # loss1 = criterion1(y1,label)
#             loss2 = criterion(y1,label)
#             loss = loss2
#             # print(loss)
#             # if batch_idx %5==0:
#             #     print(loss)
#             # print(loss)
#             # time1 = time.time()
#             optimizer.zero_grad()
            
#             loss.backward()
#             # print(loss)
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             # time2 = time.time()
#             time_bend =time.time()
#             time_save[2]+=(time_bend-time_b)
#             # time2 = time.time()
#             # print("gradient %f"%(time2-time1))
#             read_start =time.time()
            
            
#             save_1.append(loss.item())
#             # if (batch_idx+1)%1000==0:
#         end_time = time.time()
#         # print("time: %f"%(end_time-start))
#         # print("read time %f", time_save)
#         torch.save(model.module.state_dict(), path+"/"+"modelnewnewnew"+str(epoch)+".pth")
#         print(np.mean(save_1))
#         model.eval()
#         for batch_idx, batchs in enumerate(dataLoaderV):
#             InputBatch = []
#             for b in batchs:
#                 InputBatch+=b
#             y,label,y1= model(InputBatch)
#             y = y.view(-1,10,128)
#             # # y1 = y1.view(-1,1)
#             y1 = torch.squeeze(y1,1)
#             y1 = y1.view(-1,10,1)
#             # label = label.view(-1,3)
#             # print(y1.shape)
#             # loss1 = criterion1(y1,label)
            
#             loss1 = criterion(y1,label)
#             loss = loss1
#             # print(loss.item())
#             # optimizer.zero_grad()
            
#             # loss.backward()

#             # optimizer.step()
#             save_2.append(loss.item())
        
#         # torch.save(model.module.state_dict(), path+"/"+modelnewnewnew"+str(epoch)+".pth")
#         # scheduler.step(np.sqrt(np.mean(save_1)))
#         print("valid")
#         print(np.mean(save_2))
#         scheduler.step()
