import torch
import yaml
import random
import sys
sys.path.append('../')
from ATGO.dataloader import train_valid_test_iemocap_dataloader
from torch import nn, optim
import os
from pathlib import Path
import ATGO.baseline.Mutils # 双模态，单模态
import datetime
from ATGO.baseline.Mmodel import AudioEncoder,TextEncoder,SharedHead # 双模态
# from ATGO.baseline.Tmodel import SharedHead # 文本模态
# from ATGO.baseline.Amodel import SharedHead # 语音模态
import logging
import numpy as np
logger = logging.getLogger('IEMOCAP_exp')
logger.setLevel(logging.DEBUG)

def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)  # 为NumPy设置随机种子
    random.seed(seed)  # 为Python内置的random模块设置随机种子
    torch.backends.cudnn.deterministic = True  # 保证每次运行的结果是确定的
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN的自动优化

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open( "config.yml", 'r', encoding = "UTF-8" ) as f:
        cfg = yaml.load( f, Loader = yaml.FullLoader )

    set_seed(cfg["exp"]["seed"])
    sessions = [1,2,3,4,5]
    audio_dir = cfg["exp"]["audio_dir"]
    text_dir = cfg["exp"]["text_dir"]
    iemocap_csv = cfg["exp"]["iemocap_csv"]
    test_wa_sum=0
    test_ua_sum=0
    test_wf1_sum=0

    current_data = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    save_dir = os.path.join(str(Path.cwd()),f"output/{current_data}/{current_time}")

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir,"main.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    model_file_path = '/mnt/cxh10/database/hanlm/SSLER/ATGO/baseline/Mmodel.py'
    # model_file_path = '/mnt/cxh10/database/hanlm/SSLER/ATGO/baseline/Tmodel.py'
    # model_file_path = '/mnt/cxh10/database/hanlm/SSLER/ATGO/baseline/Amodel.py'
    with open(model_file_path, 'r') as file:
        file_contents = file.read()
        logger.info(file_contents)
    logger.info(cfg)

    for fold in sessions:
        logger.info(f"------Now it's {fold}th fold------")
        test_sess = fold
        batch_size = cfg["train"]["batch_size"]
        num_sess = 5
        train_loader, val_loader, test_loader = train_valid_test_iemocap_dataloader(audio_dir,text_dir,test_sess,batch_size,iemocap_csv,num_sess)
        audio_encoder = AudioEncoder(cfg).to(device)
        text_encoder = TextEncoder(cfg).to(device)
        shared_head = SharedHead(cfg).to(device)
        logger.info(audio_encoder)
        logger.info(text_encoder)
        logger.info(shared_head)
        
        optimizer = optim.RMSprop(list(audio_encoder.parameters()) + list(text_encoder.parameters()) + list(shared_head.parameters()), lr=cfg["train"]["optimizer"]["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epoch"])#余弦退火
        test_wa,test_ua,test_wf1 = ATGO.baseline.Mutils.train_one_fold_frame( logger,save_dir,fold,[audio_encoder, text_encoder, shared_head], train_loader,val_loader,test_loader,
                            optimizer, scheduler, device,cfg,continue_training=False)
        
        test_wa_sum += test_wa
        test_ua_sum += test_ua
        test_wf1_sum += test_wf1
    logger.info(f"Average WA: {test_wa_sum/len(sessions)}%; UA: {test_ua_sum/len(sessions)}%; W-F1: {test_wf1_sum/len(sessions)}%")



if __name__ == '__main__':
    main()