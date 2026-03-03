import os
import torch
import numpy as np
import random
from scipy.stats import spearmanr, pearsonr, kendalltau
import scipy.stats as stats

from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
from basicsr.models.archs.restormer_arch import Restormer
from timm.utils.inference_process import RandCrop, ToTensor, Normalize, five_point_crop, sort_file
from data.UWIQA.uwiqa_test import UWIQA
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_epoch(config, net, test_loader):
    with torch.no_grad():
        net.eval()
        name_list = []
        pred_list = []
        pred_epoch = []
        labels_epoch = []


        with open(config.valid_path + '/output.txt', 'w') as f:
            for data in tqdm(test_loader):
                pred = 0
                for i in range(config.num_avg_val):
                    x_d = data['d_img_org'].cuda()
                   # print(data)
                    labels = data['score']
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    x_d = five_point_crop(i, d_img=x_d, config=config)
                    pred += net(x_d)

                pred /= config.num_avg_val
                d_name = data['d_name']
                pred = pred.cpu().numpy()
                name_list.extend(d_name)
                pred_list.extend(pred)
                pred_batch_numpy = pred
                labels_batch_numpy = labels.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        # Calculate metrics
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_k, _ = kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        rmse = np.sqrt(np.mean(pred_epoch - labels_epoch) ** 2)


        print(f"SRCC: {rho_s}, PLCC: {rho_p}, KRCC: {rho_k}, RMSE: {rmse}")

        return rho_s, rho_p, rho_k, rmse


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
        "db_name": "UWIQA",
        "test_dis_path": "/home/adi/csc/IQAdataset/UWIQA/UWIQA",
        "uwiqa_train_label": "/home/adi/csc/pythonProject/data/UWIQA/uwiqa.txt",

        # optimization
        "batch_size": 1,
        "num_avg_val": 1,
        "crop_size": 128,

        # model
        "patch_size": 8,
        "img_size": 128,
        "inp_channels": 3,
        "dim": 48,
        "embed_dim": 32,
        "num_blocks": [2, 2, 2, 2],
        "heads": [1, 2, 4, 8],
        "num_outputs": 1,
        "drop": 0.1,
        "ffn_expansion_factor": 2.66,
        "bias": False,
        "LayerNorm_type": 'WithBias',
        "dual_pixel_task": False,

        # device
        "num_workers": 0,
        "val_keep_ratio": 0.2,

        # load & save checkpoint
        "valid": "/home/adi/csc/pythonProject/output/valid",
        "valid_path": "/home/adi/csc/pythonProject/output/valid/inference_valid",
        "model_save_path": "/home/adi/csc/pythonProject/checkpoint/uwiqa/uwiqa-base15_s20/epoch1477.pt"
    })

    if not os.path.exists(config.valid):
        os.mkdir(config.valid)

    if not os.path.exists(config.valid_path):
        os.mkdir(config.valid_path)

    # data load
    test_dataset = UWIQA(
        dis_path=config.test_dis_path,
        txt_file_name=config.uwiqa_train_label,
        transform=transforms.Compose([RandCrop(patch_size=config.crop_size),Normalize(0.5, 0.5), ToTensor()]),
        keep_ratio=config.val_keep_ratio
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False

    )
    net = Restormer(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim=config.dim,
                    drop=config.drop, ffn_expansion_factor=config.ffn_expansion_factor,
                    inp_channels=config.inp_channels,
                    bias=config.bias, heads=config.heads, num_blocks=config.num_blocks,
                    LayerNorm_type=config.LayerNorm_type)
    checkpoint = torch.load(config.model_save_path)
    net.load_state_dict(torch.load(config.model_save_path), strict=False)
    net = net.cuda()
    # net.torch.load("E://csc/pythonProject/checkpoint/uiqa/uiqa-base_s20/epoch19.pt")
    # #net = net.load_state_dist(torch.load("E://csc/pythonProject/checkpoint/uiqa/uiqa-base_s20/epoch19.pt"))
    # net = net.cuda
    losses, scores = [], []
    eval_epoch(config, net, test_loader)
    sort_file(config.valid_path + '/output.txt')

