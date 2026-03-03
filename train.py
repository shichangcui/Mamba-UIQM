import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from data.UWIQA.uwiqa import UWIQA
from data.USRD.usrd import USRD
from data.UID2021.uid2021 import UID2021
from data.UIED.UIED import UIED
from ATUIQP import  ATUIQP
from basicsr.models.archs.restormer_arch import Restormer
from timm.utils.process import RandCrop, ToTensor, Normalize, five_point_crop
from timm.utils.process import split_dataset_kadid10k, split_dataset_koniq10k
from timm.utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr, kendalltau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.use_deterministic_algorithms(False)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for data in tqdm(train_loader):
        x_d = data['d_img_org'].cuda()
       # print(x_d.shape)
        labels = data['score'].cuda()
      #  print((labels.shape))

        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
       # print(labels.shape)
        pred_d = net(x_d)
       # pred_d = F.adaptive_avg_pool2d(pred_d, (1,1))
      #  print("pred_d=", pred_d.shape)

        optimizer.zero_grad()
        inputs = torch.squeeze((pred_d))
       #  inputs = F.adaptive_avg_pool2d(inputs, (1, 1))
      #  print(inputs.shape)
        target = labels
        loss = criterion(inputs, target)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
     #   print("pred_batch_numpy=", pred_batch_numpy.shape)
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
     #   print("pred_epoch=", pred_epoch.shape)
     #   print("labels_epoch=",labels_epoch.shape)

     # Calculate RMSE
   # rmse = np.sqrt(np.mean((pred_epoch - labels_epoch) ** 2))

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_k, _ = kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}/ KRCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p, rho_k))
    print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}/ KRCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p, rho_k))

    return ret_loss, rho_s, rho_p, rho_k


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            pred = 0
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                x_d = five_point_crop(i, d_img=x_d, config=config)
                pred += net(x_d)

            pred /= config.num_avg_val
            # compute loss
            input = torch.squeeze((pred))
            target = labels
            loss = criterion(input, target)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # Calculate RMSE
      #  rmse = np.sqrt(np.mean((pred_epoch - labels_epoch) ** 2))

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_k, _ = kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info(
            'test epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}====KRCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p, rho_k))

        print(
            'test epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}=====KRCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p, rho_k))
        return np.mean(losses), rho_s, rho_p, rho_k


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "dataset_name": "UWIQA",

        # UWIQA
        "train_dis_path": "../IQA-dataset/UWIQA",
        "val_dis_path": "../IQA-dataset/UWIQA",
        "uiqa_train_label": "./data/UWIQA/uwiqa_train.txt",
        "uiqa_val_txt_label": "./data/UWIQA/uwiqa_test.txt",

        # optimization
        "batch_size": 1,
        "learning_rate": 4e-4,
        "weight_decay":4e-4,
        "n_epoch": 300,
        "val_freq": 1,
        "T_max": 20,
        "eta_min": 0,
        "num_avg_val": 1,  # if training koniq10k, num_avg_val is set to 1
        "num_workers": 8,

        # data
        "split_seed": 20,
        "train_keep_ratio": 1.0,
        "val_keep_ratio": 1.0,
        "crop_size": 224,
        "prob_aug": 0.7,

        # model
        "patch_size": 8,
        "img_size": 224,
        "inp_channels":3,
        "dim": 24,
        "embed_dim": 192,
       # "embed_dim1":384,
        "add_historgram": True,
        "his_channel": 48,
        "hist_dim": 16*3,
        "num_blocks": [2, 2, 2, 2],
        "nums": [2, 2, 2, 2],
        "num_outputs": 1,
        "drop": 0.1,
        "ffn_expansion_factor": 2.66,
        "bias": False,
        "LayerNorm_type": 'WithBias',
        "dual_pixel_task": False,
        "res" : 128,
        "infer_mode": False,
        "hidden_dim": 48,
        "headdim": 12,
        "drop_path": 0.1,
        "use_checkpoint": False,
        "act_layer": nn.GELU,
        "mlp_ratio": 2.0,
        "post_norm": True,
        "layer_scale": None,

        # load & save checkpoint
        "model_name": "uwiqa-base(融合)_s20",
        "type_name": "uwiqa",
        "ckpt_path": "./checkpoint/",  # directory for saving checkpoint
        "log_path": "./checkpoint/log/",
        "log_file": ".log",
        "train_continue": False,
        "model_save_path": "./checkpoint/uwiqa/uwiqa-base(融合)_s20/epoch208.pt"
    })

    config.log_file = config.model_name + ".log"

    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)

    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.log_path)



    # data load
    train_dataset = UWIQA(
        dis_path=config.train_dis_path,
        txt_file_name=config.uiqa_train_label,
        transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
                                      Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=config.prob_aug), ToTensor()]),
        keep_ratio=config.train_keep_ratio
    )

    val_dataset = UWIQA(
        dis_path=config.val_dis_path,
        txt_file_name=config.uiqa_val_txt_label,
        transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
                                      Normalize(0.5, 0.5),  ToTensor()]),
        keep_ratio=config.val_keep_ratio
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    # load the data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              num_workers=config.num_workers, drop_last=True, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, drop_last=True, shuffle=False)

    # model defination
    net = Restormer(embed_dim=config.embed_dim,  num_outputs=config.num_outputs, dim=config.dim, res=config.res, hidden_dim=config.hidden_dim,drop_path=config.drop_path,
                mlp_ratio=config.mlp_ratio, post_norm=config.post_norm, layer_scale=config.layer_scale, drop=config.drop, ffn_expansion_factor=config.ffn_expansion_factor, inp_channels=config.inp_channels,
                 bias=config.bias, nums=config.nums, num_blocks=config.num_blocks, LayerNorm_type=config.LayerNorm_type, headdim=config.headdim)

#    net = ATUIQP()
    epoch_num = -1
    if config.train_continue:
        epoch_num = config.model_save_path.split('/')[-1]
        epoch_num = epoch_num.split('.')[0]
        epoch_num = int(epoch_num[5:])
        # ccheckpoint = torch.load(config.model_save_path)
        net.load_state_dict(torch.load(config.model_save_path), strict=False)
    net = net.cuda()

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    net = nn.DataParallel(net)
    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
   # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    best_krcc = 0
    main_score = 0
    for epoch in range(0, config.n_epoch):
        if epoch < epoch_num:
            continue
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p, rho_k = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)
        writer.add_scalar("KRCC", rho_k, epoch)


        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p, rho_k = eval_epoch(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

            if rho_s + rho_p +rho_k > main_score:
                main_score = rho_s + rho_p + rho_k
                best_srocc = rho_s
                best_plcc = rho_p
                best_krcc = rho_k


                logging.info('=====================best SRCC is{}================================================='.format(best_srocc))
                logging.info(
                    '============================== best main score is {} ================================='.format(
                        main_score))
                logging.info('==========================best PLCC is {}=================================================='.format(best_plcc))
                logging.info(
                    '==========================best KRCC is {}=================================================='.format(
                        best_krcc))

                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.ckpt_path, model_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info(
                    'Saving weights and model of epoch{}, SRCC:{}, PLCC:{}, KRCC:{}'.format(epoch + 1, best_srocc, best_plcc, best_krcc))

        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

