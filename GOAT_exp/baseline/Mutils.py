import torch
from torch import nn, optim
import os
from pathlib import Path
from time import time
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support


def evaluate_accuracy_frame(audio_encoder, text_encoder,shared_head, val_loader, device):
    """验证模型预测结果的WA、UA与混淆矩阵"""
    shared_head.eval()
    num_val = len(val_loader.dataset)
    acc = 0
    mat = np.zeros((4, 4))  # 混淆矩阵
    class_num = np.zeros(4)
    acc_num = np.zeros(4)
    all_preds = []
    all_labels = []
    weight_audio,weight_text = 0.5,0.5
    with torch.no_grad():
        for batch in val_loader:
            ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
            text_feat = net_input["text_feats"].to(device)
            audio_feat = net_input["audio_feats"].to(device)
            true = int(labels.item())
            class_num[true] += 1
            text_encoded = text_encoder(text_feat)
            audio_encoded = audio_encoder(audio_feat)
            out_t,_ = shared_head(text_encoded)
            out_a,_ = shared_head(audio_encoded)
            output = (out_a * weight_audio + out_t * weight_text)
            pred = int(torch.argmax(output, dim=-1).item())
            all_preds.append(pred)
            all_labels.append(true)
            mat[true, pred] += 1
            if true == pred:
                acc += 1
                acc_num[pred] += 1
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(4)), zero_division=0)
    wf1 = np.average(f1, weights=class_num)
    # 返回WA, UA, confusion matrix
    return acc / num_val, np.mean(acc_num / class_num), wf1, mat

# 验证集损失
def evaluate_loss_frame(audio_encoder, text_encoder,shared_head, val_loader, device):
    """计算验证集上的损失"""
    audio_encoder.eval()
    text_encoder.eval()
    shared_head.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    weight_audio,weight_text = 0.5,0.5
    with torch.no_grad():
        for batch in val_loader:
            ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
            text_feat = net_input["text_feats"].to(device)
            audio_feat = net_input["audio_feats"].to(device)
            labels = labels.to(device)
            text_encoded = text_encoder(text_feat)
            audio_encoded = audio_encoder(audio_feat)
            out_t,_ = shared_head(text_encoded)
            out_a,_ = shared_head(audio_encoded)
            output = (out_a * weight_audio + out_t * weight_text)
            loss = criterion(output, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_one_fold_frame(logger, save_dir, fold, models, train_loader, val_loader, test_loader,
                         optimizer, scheduler, device, cfg, continue_training=False):
    """模型训练，在每个epoch报道验证集和验证集的结果,在每折报告测试的结果"""
    audio_encoder, text_encoder, shared_head = models
    best_val_wa = 0
    best_val_wa_ua = 0  # wa最大时的ua
    best_val_wa_epoch = 0
    save_dir = os.path.join(str(Path.cwd()), f"{save_dir}/model_{fold}.pth")
    num_epochs = cfg["train"]["epoch"]
    # gradient_modifier = GradientModifier()

    # 如果继续训练，加载之前保存的最佳模型和优化器状态
    if continue_training and os.path.exists(save_dir):
        checkpoint = torch.load(save_dir)
        audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        shared_head.load_state_dict(checkpoint['shared_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_wa = checkpoint['best_val_wa']
        best_val_wa_ua = checkpoint['best_val_wa_ua']
        best_val_wa_epoch = checkpoint['best_val_wa_epoch']
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Continuing training from epoch {start_epoch}")
    else:
        start_epoch = 1

    for epoch in range(start_epoch, num_epochs + 1):
        audio_encoder.train()
        text_encoder.train()
        shared_head.train()
        train_l_sum, train_acc_sum = 0, 0 # 损失与准确率的和
        start = time()
        weight_audio,weight_text = 0.5,0.5
        for batch_index, batch in enumerate(train_loader):
            ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
            text_feat = net_input["text_feats"].to(device)
            audio_feat = net_input["audio_feats"].to(device)
            labels = labels.to(device)
            text_encoded = text_encoder(text_feat)
            audio_encoded = audio_encoder(audio_feat)
            out_t,_ = shared_head(text_encoded)
            out_a,_ = shared_head(audio_encoded)
            output = (out_a * weight_audio + out_t * weight_text)
            loss = nn.CrossEntropyLoss()(output, labels)
            optimizer.zero_grad()
            if np.isnan( loss.item() ): # 发散，应该终止训练
                logger.info( "Error: loss diverges." )
                return
            loss.backward()
            # text_grad_list.append(torch.norm(text_feat.grad, p=2).item())
            # audio_grad_list.append(torch.norm(audio_feat.grad, p=2).item())
            optimizer.step()
            train_l_sum += loss.item()
            acc_num = ( torch.argmax( output, dim = 1 ) == labels ).sum().item()
            train_acc_sum += acc_num
        #         # 计算整个训练过程的平均梯度
        # logger.info(f"Avg Text Gradient: {sum(text_grad_list) / len(text_grad_list):.6f}")
        # logger.info(f"Avg Audio Gradient: {sum(audio_grad_list) / len(audio_grad_list):.6f}")
        
        
        if scheduler:  # 在每个epoch结束后进行更新
            scheduler.step()
        train_acc_sum /= len( train_loader.dataset )
        val_wa, val_ua, val_wf1,mat = evaluate_accuracy_frame( audio_encoder, text_encoder, shared_head, val_loader, device )
        if (val_wa > best_val_wa) or (val_wa == best_val_wa and val_ua > best_val_wa_ua):
            best_val_wa = val_wa
            best_val_wa_epoch = epoch
            best_val_wa_ua = val_ua
            torch.save({
                'epoch': epoch,
                'audio_encoder_state_dict': audio_encoder.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'shared_head_state_dict': shared_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_wa': best_val_wa,
                'best_val_wa_ua': best_val_wa_ua,
                'best_val_wa_epoch': best_val_wa_epoch
            }, save_dir)

        # 计算验证集损失
        val_loss = evaluate_loss_frame(audio_encoder, text_encoder, shared_head, val_loader, device)

        # 记录每一次的WA, UA与混淆矩阵
        logger.info( 'epoch %d loss %.4f train_acc %.2f time %.2f s'
              % ( epoch, train_l_sum / len( train_loader ), train_acc_sum * 100, time() - start ) )
        logger.info( 'val_wa %f  val_ua %f val_wf1 %f val_loss %.2f' 
              % ( val_wa*100, val_ua*100, val_wf1*100, val_loss ) )
        logger.info(mat)  

    checkpoint = torch.load(save_dir)
    shared_head.load_state_dict(checkpoint['shared_head_state_dict'], strict=True)
    test_wa, test_ua, test_wf1, mat = evaluate_accuracy_frame(audio_encoder, text_encoder,shared_head, test_loader, device)
    logger.info(f"The {fold}th Fold at epoch {best_val_wa_epoch},test_wa {test_wa * 100} test_ua {test_ua * 100},test_wf1 {test_wf1 * 100}")
    return test_wa * 100, test_ua * 100, test_wf1 * 100