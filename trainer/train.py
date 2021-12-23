import gc
from utils.metrics import mean_average_precision_impression
from utils.mylib import *
from utils.loss import *
import os
from utils.visualize import *
from dataset import *
import pandas as pd
def gan_train(param):
    # paramの変数
    epoch = param['epoch']
    opts = param["opts"]
    G_model = param["G_model"]
    D_model = param["D_model"]
    fid = param['fid']
    mAP_score = param['mAP_score']
    DataLoader = param["DataLoader"]
    ID = param['ID']
    test_z = param["z"]
    G_lr_scheduler = param["G_lr_scheduler"]
    D_lr_scheduler = param["D_lr_scheduler"]
    G_optimizer = param["G_optimizer"]
    D_optimizer = param["D_optimizer"]
    iter_start = param["iter_start"]
    writer = param['writer']
    ##training start
    G_model.train()
    D_model.train()
    iter = iter_start
    #lossの初期化
    D_running_TF_loss = 0
    G_running_TF_loss = 0
    D_running_cl_loss = 0
    G_running_cl_loss = 0
    real_acc = []
    fake_acc = []
    #Dataloaderの定義
    databar = tqdm.tqdm(DataLoader)

    last_activation = nn.Softmax(dim=1)
    imp_loss = KlLoss(activation='softmax').to(opts.device)
    #mAPを記録するためのリスト
    prediction_imp = []
    target_imp = []
    G_model.train()
    D_model.train()
    for batch_idx, samples in enumerate(databar):
        real_img, char_class, labels \
            = samples['img'], samples['charclass'], samples['embed_label']
        # バッチの長さを定義
        batch_len = real_img.size(0)
        #デバイスの移
        real_img,  char_class = \
            real_img.to(opts.device), char_class.to(opts.device)
        # 文字クラスのone-hotベクトル化
        char_class_oh = torch.eye(opts.char_num)[char_class].to(opts.device)
        # 印象語のベクトル
        labels_oh = Multilabel_OneHot(labels, len(ID), normalize=True).to(opts.device)
        # training Generator
        for _ in range(opts.num_critic):
            #画像の生成に必要なノイズ作成
            z = torch.normal(mean = 0.5, std = 0.2, size = (batch_len, opts.latent_size)).to(opts.device)
            ##画像の生成に必要な印象語ラベルを取得
            _,  D_real_class = D_model(real_img, char_class_oh)
            gen_label = last_activation(D_real_class.detach()).to(opts.device)
            # ２つのノイズの結合
            fake_img, _ = G_model(z, char_class_oh, gen_label)
            D_fake_TF, D_fake_class = D_model(fake_img, char_class_oh)
            # Wasserstein lossの計算
            G_TF_loss = -torch.mean(D_fake_TF)
            # 印象語分類のロス
            G_class_loss = imp_loss(D_fake_class, gen_label)
            # mode seeking lossの算出
            G_loss = G_TF_loss + G_class_loss * opts.lambda_class
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            gc.collect()
            G_running_TF_loss += G_TF_loss.item()
            G_running_cl_loss += G_class_loss.item()

        #training Discriminator
        #Discriminatorに本物画像を入れて順伝播⇒Loss計算
        # 生成用のラベル
        fake_img, _ = G_model(z, char_class_oh, gen_label)
        D_real_TF, D_real_class = D_model(real_img, char_class_oh)
        D_real_loss = -torch.mean(D_real_TF)
        D_fake, _ = D_model(fake_img.detach(), char_class_oh)
        D_fake_loss = torch.mean(D_fake)
        gp_loss = compute_gradient_penalty(D_model, real_img.data, fake_img.data, opts.Tensor, char_class=char_class_oh)
        loss_drift = (D_real_TF ** 2).mean()
        ## scに関する一貫性損失
        #Wasserstein lossの計算
        D_TF_loss = D_fake_loss + D_real_loss + opts.lambda_gp * gp_loss
        # 印象語分類のロス
        D_class_loss = imp_loss(D_real_class, labels_oh)
        D_loss = 0.1 * D_TF_loss + D_class_loss + 0.001 * loss_drift

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        D_running_TF_loss += D_TF_loss.item()
        D_running_cl_loss += D_class_loss.item()
        real_pred = 1 * (torch.sigmoid(D_real_TF) > 0.5).detach().cpu()
        fake_pred = 1 * (torch.sigmoid(D_fake) > 0.5).detach().cpu()
        real_TF = torch.ones(real_pred.size(0))
        fake_TF = torch.zeros(fake_pred.size(0))
        r_acc = (real_pred == real_TF).float().sum().item() / len(real_pred)
        f_acc = (fake_pred == fake_TF).float().sum().item() / len(fake_pred)
        real_acc.append(r_acc)
        fake_acc.append(f_acc)
        prediction_imp.append(D_real_class.detach().cpu())
        target_imp.append(labels_oh.detach().cpu())

        ##tensor bord
        writer.add_scalars("TF_loss", {'D_TF_loss': D_TF_loss, 'G_TF_loss': G_TF_loss}, iter)
        writer.add_scalars("class_loss", {'D_class_loss': D_class_loss, 'G_class_loss': G_class_loss}, iter)
        writer.add_scalars("Acc", {'real_acc': r_acc, 'fake_acc': f_acc}, iter)
        G_lr_scheduler.step()
        D_lr_scheduler.step()
        iter += 1

        if iter % 200 == 0:
            test_label = ['decorative', 'big', 'shade', 'manuscript', 'ghost']
            test_emb_label = [[ID[key]] for key in test_label]
            label = Multilabel_OneHot(test_emb_label, len(ID), normalize=False)
            save_path = os.path.join(opts.logs_GAN, 'img_iter_%05d_%02d✕%02d.png' % (iter, real_img.size(2), real_img.size(3)))
            visualizer(save_path, G_model, test_z, opts.char_num, label, opts.device)
        if iter % 2000 == 0:
            weight = {'G_net': G_model.state_dict(),
                   'G_optimizer': G_optimizer.state_dict(),
                   'D_net': D_model.state_dict(),
                   'D_optimizer': D_optimizer.state_dict()}
            torch.save(weight, os.path.join(opts.weight_dir, 'weight_iter_%d.pth' % (iter)))
        if iter == 90000:
            break
    prediction_imp = torch.cat(prediction_imp, axis=0)
    target_imp = torch.cat(target_imp, axis=0)
    _, score = mean_average_precision_impression(prediction_imp, target_imp)
    mAP_score.loc[f"epoch_{epoch}"] = score
    fid_disttance = fid.calculate_fretchet(real_img.data.cpu().repeat(1, 3, 1, 1),
                                           fake_img.data.cpu().repeat(1, 3, 1, 1),  cuda=opts.device, batch_size=opts.batch_size//4)
    writer.add_scalar("fid", fid_disttance, epoch)
    D_running_TF_loss /= len(DataLoader)
    G_running_TF_loss /= len(DataLoader)
    D_running_cl_loss /= len(DataLoader)
    G_running_cl_loss /= len(DataLoader)
    real_acc = sum(real_acc)/len(real_acc)
    fake_acc = sum(fake_acc)/len(fake_acc)
    check_point = {
                   'D_epoch_TF_losses': D_running_TF_loss,
                   'G_epoch_TF_losses': G_running_TF_loss,
                   'D_epoch_cl_losses': D_running_cl_loss,
                   'G_epoch_cl_losses': G_running_cl_loss,
                    'FID': fid_disttance,
                    "mAP_score": mAP_score,
                   'epoch_real_acc': real_acc,
                  'epoch_fake_acc':fake_acc,
                   "iter_finish": iter,
                   }
    return check_point