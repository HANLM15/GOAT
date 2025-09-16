import torch
from torch import nn, optim
import os
from pathlib import Path
from time import time
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

def evaluate_accuracy_frame(audio_encoder, text_encoder, shared_head, val_loader, device):
    """验证模型预测结果的WA、UA与混淆矩阵"""
    audio_encoder.eval()
    text_encoder.eval()
    shared_head.eval()
    num_val = len(val_loader.dataset)
    acc = 0
    mat = np.zeros((4, 4))  # 混淆矩阵
    class_num = np.zeros(4)
    acc_num = np.zeros(4)
    all_preds = []
    all_labels = []
    # total_loss = 0
    # criterion = nn.CrossEntropyLoss()
    weight_audio,weight_text = 0.5,0.5
    with torch.no_grad():
        for batch in val_loader:
            ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
            text_feat = net_input["text_feats"].to(device)
            audio_feat = net_input["audio_feats"].to(device)
            text_padding_mask = net_input["text_padding_mask"].to(device)
            audio_padding_mask = net_input["audio_padding_mask"].to(device)
            true = int(labels.item())
            class_num[true] += 1
            text_encoded = text_encoder(text_feat)
            audio_encoded = audio_encoder(audio_feat)
            out_t,_ = shared_head(text_encoded)
            out_a,_ = shared_head(audio_encoded)
            
            # weight_audio,weight_text = calculate_gating_weights(out_a, out_t)
            output = (out_a * weight_audio + out_t * weight_text)
            pred = int(torch.argmax(output, dim=-1).item())
            all_preds.append(pred)
            all_labels.append(true)
            mat[true, pred] += 1
            if true == pred:
                acc += 1
                acc_num[pred] += 1
            # loss = criterion(output, labels).to(device)
            # total_loss += loss.item()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(4)), zero_division=0)
    wf1 = np.average(f1, weights=class_num)
    # 返回WA, UA, confusion matrix
    return acc / num_val, np.mean(acc_num / class_num), wf1, mat

def evaluate_accuracy_single_modality(encoder, shared_head, val_loader, device, modality_name):
    encoder.eval()
    shared_head.eval()
    acc = 0
    mat = np.zeros((4, 4))
    class_num = np.zeros(4)
    acc_num = np.zeros(4)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            net_input = batch["net_input"]
            labels = batch["labels"].to(device)
            if modality_name == "text":
                feats = net_input["text_feats"].to(device)
            elif modality_name == "audio":
                feats = net_input["audio_feats"].to(device)
            else:
                raise ValueError("Unknown modality")

            encoded = encoder(feats)
            output, _ = shared_head(encoded)

            pred = output.argmax(dim=1)
            for t, p in zip(labels, pred):
                class_num[t.item()] += 1
                acc_num[p.item()] += (p == t).item()
                mat[t.item(), p.item()] += 1
                if p.item() == t.item():
                    acc += 1
                all_preds.append(p.item())
                all_labels.append(t.item())

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(4)), zero_division=0)
    wf1 = np.average(f1, weights=class_num)
    wa = acc / len(val_loader.dataset)
    ua = np.mean(acc_num / class_num)
    return wa, ua, wf1, mat,encoded


# 验证集损失
def evaluate_loss_frame(audio_encoder, text_encoder, shared_head, val_loader, device):
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
            text_padding_mask = net_input["text_padding_mask"].to(device)
            audio_padding_mask = net_input["audio_padding_mask"].to(device)
            labels = labels.to(device)
            text_encoded = text_encoder(text_feat)
            audio_encoded = audio_encoder(audio_feat)
            out_t,_ = shared_head(text_encoded)
            out_a,_ = shared_head(audio_encoded)
            
            # weight_audio,weight_text = calculate_gating_weights(out_a, out_t)
            output = (out_a * weight_audio + out_t * weight_text)
            loss = criterion(output, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

class GM:
    def __init__(self,device):
        # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # self.P1 = torch.eye(256).type(dtype).to(device)
        # self.P2 = torch.eye(64).type(dtype).to(device)
        self.exp_count = 0

    def before_update(self, logger, model, before_batch_input,fc1_out, batch_index, len_dataloader, train_exp_counter):
        # epsilon = 1e-6  # 避免除 0
        epsilon = 0 
        if train_exp_counter != 0:
            for n, w in model.named_parameters():
                # if w.grad is not None:
                #     logger.info(f"Layer: {n}")
                #     logger.info(f'grad max: {w.grad.data.max()}')
                #     logger.info(f'grad min: {w.grad.data.min()}')

                if "st" in n and "weight" in n:
                    # 计算输入的均值
                    r = torch.mean(before_batch_input, dim=0, keepdim=True) #[1, 256]
                    r = r.squeeze(0)
                    # 计算正交投影
                    norm_r = torch.norm(r)**2  # 避免除 0 ,tensor(4.7303
                    delta_w = w.grad # [64, 256]
                    projection_coefficient = torch.matmul(delta_w, r) / norm_r  # [64, 1]
                    projection_vector = projection_coefficient.unsqueeze(1) *r.unsqueeze(0)
                    delta_w = delta_w - projection_vector
                    res = torch.matmul(delta_w,r)
                    w.grad = delta_w
                elif "c_fc" in n and "weight" in n:
                   # 计算输入的均值
                    r = torch.mean(fc1_out, dim=0, keepdim=True) # [1, 64]
                    r = r.squeeze(0)
                    norm_r = torch.norm(r)**2  # 避免除 0 ,tensor(4.7303
                    delta_w = w.grad # [64, 256]
                    projection_coefficient = torch.matmul(delta_w, r) / norm_r  # [64, 1]
                    projection_vector = projection_coefficient.unsqueeze(1) *r.unsqueeze(0)
                    delta_w = delta_w - projection_vector
                    res = torch.matmul(delta_w,r)
                    w.grad = delta_w
                # if "st" in n and "weight" in n:
                #     # 计算输入的均值
                #     r = torch.mean(before_batch_input, dim=0, keepdim=True) #[1, 256]
                #     # 计算正交投影
                #     norm_r = torch.norm(r, p=2) + epsilon  # 避免除 0 ,tensor(4.7303
                #     projection = torch.mm(self.P1, torch.t(r)) / norm_r
                #     self.P1 = self.P1 - projection * r 
                #     pnorm2 = torch.norm(self.P1, p='fro')
                #     self.P1.data = self.P1.data / (pnorm2 + epsilon)
                #     w.grad.data = torch.mm(w.grad.data, torch.t(self.P1.data))
                # elif "c_fc" in n and "weight" in n:
                #    # 计算输入的均值
                #     r = torch.mean(fc1_out, dim=0, keepdim=True) 
                #     norm_r = torch.norm(r, p=2) + epsilon  # 避免除 0 ,tensor(4.7303
                #     projection = torch.mm(self.P2, torch.t(r)) / norm_r
                #     self.P2 = self.P2 - projection * r 
                #     pnorm2 = torch.norm(self.P2, p='fro')
                #     self.P2.data = self.P2.data / (pnorm2 + epsilon)
                #     w.grad.data = torch.mm(w.grad.data, torch.t(self.P2.data))

class GM_B:
    def __init__(self):
        self.exp_count = 0

    def before_update(self, epoch, model, before_batch_input,fc1_out, device, len_dataloader, train_exp_counter):
        # epsilon = 1e-6  # 避免除 0
        epsilon = 0 
        if train_exp_counter != 0:
            for n, w in model.named_parameters():
                # if w.grad is not None:
                #     logger.info(f"Layer: {n}")
                #     logger.info(f'grad max: {w.grad.data.max()}')
                #     logger.info(f'grad min: {w.grad.data.min()}')
                if epoch > 5:
                    if "st" in n and "bias" in n:
                        w.grad = torch.zeros(w.grad.shape).to(device)
                    elif "c_fc" in n and "bias" in n:
                        w.grad = torch.zeros(w.grad.shape).to(device)
                if "st" in n and "weight" in n:
                    # 计算输入的均值
                    r = torch.mean(before_batch_input, dim=0, keepdim=True) #[1, 256]
                    r = r.squeeze(0)#[256]
                    # 计算正交投影
                    norm_r = torch.norm(r)**2  # 避免除 0 ,tensor(4.7303
                    delta_w = w.grad # [64, 256]
                    projection_coefficient = torch.matmul(delta_w, r) / norm_r  # [64]
                    projection_vector = projection_coefficient.unsqueeze(1) *r.unsqueeze(0) #[64,256]
                    delta_w = delta_w - projection_vector
                    res = torch.matmul(delta_w,r)
                    w.grad = delta_w
                elif "c_fc" in n and "weight" in n:
                   # 计算输入的均值
                    r = torch.mean(fc1_out, dim=0, keepdim=True) 
                    r = r.squeeze(0)
                    norm_r = torch.norm(r)**2  # 避免除 0 ,tensor(4.7303
                    delta_w = w.grad 
                    projection_coefficient = torch.matmul(delta_w, r) / norm_r  
                    projection_vector = projection_coefficient.unsqueeze(1) *r.unsqueeze(0)
                    delta_w = delta_w - projection_vector
                    res = torch.matmul(delta_w,r)
                    w.grad = delta_w

def train_single_modality2(epoch,gradient_modifier,encoder, shared_head, train_loader, audio_optimizer, text_optimizer, device, modality_name):
    running_loss = 0.0
    train_acc_sum = 0
    encoder.train()
    shared_head.train()
    total_batches = len(train_loader)
    # 根据模态选择优化器和相关参数
    if modality_name == "text":
        optimizer = text_optimizer
    elif modality_name == "audio":
        optimizer = audio_optimizer
    else:
        raise ValueError(f"Invalid modality_name: {modality_name}")
    
    for i, batch in enumerate(train_loader):
        ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
        text_feat = net_input["text_feats"].to(device)
        audio_feat = net_input["audio_feats"].to(device)
        labels = labels.to(device)
        if modality_name == "text":
            encoded_feat = encoder(text_feat)
            output,fc_out = shared_head(encoded_feat) #原始model
        else:
            encoded_feat = encoder(audio_feat)
            output,fc_out = shared_head(encoded_feat)

        saved_encoded_feat = encoded_feat
        loss = nn.CrossEntropyLoss()(output, labels)
        optimizer.zero_grad()
        loss.backward()
        gradient_modifier.before_update(epoch, shared_head, encoded_feat,fc_out, device, total_batches, gradient_modifier.exp_count)
        gradient_modifier.exp_count += 1 
        optimizer.step()

        running_loss += loss.item()
        acc_num = ( torch.argmax( output, dim = 1 ) == labels ).sum().item()
        train_acc_sum += acc_num

    return running_loss / len(train_loader) , train_acc_sum / len( train_loader.dataset ),saved_encoded_feat

def train_one_fold(logger, save_dir, fold, models, train_loader, val_loader, test_loader,
                         optimizers, schedulers, device, cfg, continue_training=False):
# def train_one_fold(logger, save_dir, fold, models, train_loader, val_loader, test_loader,
#                          optimizers, schedulers, device, cfg, alpha, continue_training=False):
    """模型训练，在每个epoch报道验证集和验证集的结果,在每折报告测试的结果"""
    audio_encoder, text_encoder, shared_head = models
    audio_optimizer, text_optimizer = optimizers
    audio_scheduler, text_scheduler = schedulers

    best_val_wa = 0
    best_val_wa_ua = 0  # wa最大时的ua
    best_val_wa_epoch = 0
    save_dir = os.path.join(str(Path.cwd()), f"{save_dir}/model_{fold}.pth")
    num_epochs = cfg["train"]["epoch"]
    # gradient_modifier = GM(device)
    gradient_modifier = GM_B()
    # 如果继续训练，加载之前保存的最佳模型和优化器状态
    if continue_training and os.path.exists(save_dir):
        checkpoint = torch.load(save_dir)
        audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        shared_head.load_state_dict(checkpoint['shared_head_state_dict'])
        audio_optimizer.load_state_dict(checkpoint.get('audio_optimizer_state_dict', {}))
        text_optimizer.load_state_dict(checkpoint.get('text_optimizer_state_dict', {}))
        # shared_head_optimizer.load_state_dict(checkpoint.get('shared_head_optimizer_state_dict', {}))
        best_val_wa = checkpoint['best_val_wa']
        best_val_wa_ua = checkpoint['best_val_wa_ua']
        best_val_wa_epoch = checkpoint['best_val_wa_epoch']
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Continuing training from epoch {start_epoch}")
    else:
        start_epoch = 1
    for epoch in range(start_epoch, num_epochs + 1):
        for modality_name, encoder in [("text", text_encoder), ("audio", audio_encoder)]:
            start = time()
            loss, acc, saved_encoded_feat = train_single_modality2(
                epoch, gradient_modifier, encoder, shared_head, train_loader,
                audio_optimizer, text_optimizer, device, modality_name
            )
            logger.info(f'Epoch {epoch}, Modality: {modality_name}, Acc: {acc} , Loss: {loss}')

            # 验证另一个模态
            other_modality = "audio" if modality_name == "text" else "text"
            other_encoder = audio_encoder if other_modality == "audio" else text_encoder
            val_wa_other, val_ua_other, val_wf1_other, mat_other, _ = evaluate_accuracy_single_modality(
                other_encoder, shared_head, val_loader, device, modality_name=other_modality
            )
            logger.info(f'AFTER {modality_name.upper()} TRAIN - {other_modality.upper()} VAL: '
                        f'WA {val_wa_other*100:.2f} UA {val_ua_other*100:.2f} WF1 {val_wf1_other*100:.2f}')
            logger.info(mat_other)

            # 验证当前模态
            val_wa_curr, val_ua_curr, val_wf1_curr, mat_curr, _ = evaluate_accuracy_single_modality(
                encoder, shared_head, val_loader, device, modality_name=modality_name
            )
            logger.info(f'AFTER {modality_name.upper()} TRAIN - {modality_name.upper()} VAL: '
                        f'WA {val_wa_curr*100:.2f} UA {val_ua_curr*100:.2f} WF1 {val_wf1_curr*100:.2f}')
            logger.info(mat_curr)

            # 验证融合
            val_wa_fuse, val_ua_fuse, val_wf1_fuse, mat_fuse = evaluate_accuracy_frame(
                audio_encoder, text_encoder, shared_head, val_loader, device
            )
            logger.info(f'AFTER {modality_name.upper()} TRAIN - FUSION VAL: '
                        f'WA {val_wa_fuse*100:.2f} UA {val_ua_fuse*100:.2f} WF1 {val_wf1_fuse*100:.2f}')
            logger.info(mat_fuse)

            # 模型保存
            if (val_wa_fuse > best_val_wa) or (val_wa_fuse == best_val_wa and val_ua_fuse > best_val_wa_ua):
                best_val_wa = val_wa_fuse
                best_val_wa_epoch = epoch
                best_val_wa_ua = val_ua_fuse
                torch.save({
                    'epoch': epoch,
                    'audio_encoder_state_dict': audio_encoder.state_dict(),
                    'text_encoder_state_dict': text_encoder.state_dict(),
                    'shared_head_state_dict': shared_head.state_dict(),
                    'audio_optimizer_state_dict': audio_optimizer.state_dict(),
                    'text_optimizer_state_dict': text_optimizer.state_dict(),
                    'best_val_wa': best_val_wa,
                    'best_val_wa_ua': best_val_wa_ua,
                    'best_val_wa_epoch': best_val_wa_epoch
                }, save_dir)

        # 所有模态训练结束后更新 scheduler
        text_scheduler.step()
        audio_scheduler.step()

        # 打印融合验证损失
        val_loss = evaluate_loss_frame(audio_encoder, text_encoder, shared_head, val_loader, device)
        logger.info(f'Epoch {epoch} FINAL val_loss: {val_loss:.4f}, time: {time() - start:.2f}s')

    checkpoint = torch.load(save_dir)
    audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'], strict=True)
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'], strict=True)
    shared_head.load_state_dict(checkpoint['shared_head_state_dict'], strict=True)
    test_wa, test_ua, test_wf1, mat_fuse = evaluate_accuracy_frame(audio_encoder, text_encoder, shared_head, test_loader, device)
    logger.info(f"The {fold}th Fold at epoch {best_val_wa_epoch},test_wa {test_wa * 100} test_ua {test_ua * 100},test_wf1 {test_wf1 * 100}")
    return test_wa * 100, test_ua * 100, test_wf1 * 100
