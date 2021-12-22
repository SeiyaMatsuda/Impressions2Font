from trainer.train import gan_train
from utils.mylib import *
from utils.logger import init_logger
from dataset import *
from models.model import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import FID
import pandas as pd
import torch.nn as nn
import os
import time
def make_logdir(path):
    log_dir = path
    weight_dir = os.path.join(log_dir, 'weight')
    logs_GAN = os.path.join(log_dir, "learning_image")
    learning_log_dir = os.path.join(log_dir, "learning_log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(logs_GAN):
        os.makedirs(logs_GAN)
    if not os.path.exists(learning_log_dir):
        os.makedirs(learning_log_dir)
    return log_dir, weight_dir, logs_GAN, learning_log_dir

def model_run(opts):
    data = np.array([np.load(d) for d in opts.data])
    label = opts.impression_word_list
    # 生成に必要な乱数
    z = torch.randn(4, opts.latent_size)
    #単語IDの変換
    ID = {key:idx+1 for idx, key in enumerate(opts.w2v_vocab)}
    weights = np.array(list(opts.w2v_vocab.values()))
    imp_num = weights.shape[0]
    #モデルを定義
    D_model = Discriminator(num_dimension=opts.w2v_dimension, imp_num=imp_num, char_num=opts.char_num).to(opts.device)
    G_model = Generator(weights, latent_size=opts.latent_size, num_dimension=opts.w2v_dimension, char_num=opts.char_num, normalize=True).to(opts.device)
    fid = FID()
    mAP_score = pd.DataFrame(columns=list(ID.keys()))
    LOGGER.info(f"================Generator================")
    LOGGER.info(f"{G_model}")
    LOGGER.info(f"================Discriminator================")
    LOGGER.info(f"{D_model}")

    #学習済みモデルのパラメータを使用
    # GPUの分散
    if opts.device_count > 1:
        D_model = nn.DataParallel(D_model)
        G_model = nn.DataParallel(G_model)
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=opts.d_lr, betas=(0, 0.99), eps=1e-08, weight_decay=1e-5,
                                   amsgrad=False)
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=opts.g_lr, betas=(0, 0.99), eps=1e-08, weight_decay=1e-5,
                                   amsgrad=False)
    def lr_lambda(iter):
        if opts.num_iterations_decay > 0:
            lr = 1.0 - max(0,
                           (iter + 1 -
                            (opts.num_iterations - opts.num_iterations_decay)
                            )) / float(opts.num_iterations_decay)
        else:
            lr = 1.0
        return lr
    G_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(G_optimizer,
                                                lr_lambda=lr_lambda)
    D_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(D_optimizer,
                                                lr_lambda=lr_lambda)
    D_TF_loss_list = []
    G_TF_loss_list = []
    D_cl_loss_list = []
    G_cl_loss_list = []
    real_acc_list = []
    fake_acc_list = []
    FID_score = []

    transform = Transform()
    #training param
    iter_start = opts.start_iterations
    writer = SummaryWriter(log_dir=opts.learning_log_dir)
    dataset = Myfont_dataset3(data, label, ID, char_num=opts.char_num,
                                  transform=transform)

    bs = opts.batch_size
    for epoch in range(opts.num_epochs):
        start_time = time.time()
        LOGGER.info(f"================epoch_{epoch}================")
        DataLoader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True,
                                                 collate_fn=collate_fn, drop_last=True, pin_memory=True, num_workers=4)
        param = {"opts": opts,  'epoch': epoch, 'G_model': G_model, 'D_model': D_model,
                 "dataset": dataset, "z": z, "fid": fid, "mAP_score": mAP_score,
                 "Dataset": dataset, 'DataLoader': DataLoader,
                 'G_optimizer': G_optimizer, 'D_optimizer': D_optimizer,
                 'G_lr_scheduler': G_lr_scheduler, 'D_lr_scheduler': D_lr_scheduler,
                 'log_dir': opts.logs_GAN, "iter_start":iter_start, 'ID': ID, 'writer': writer}
        check_point = gan_train(param)
        iter_start = check_point["iter_finish"]
        D_TF_loss_list.append(check_point["D_epoch_TF_losses"])
        G_TF_loss_list.append(check_point["G_epoch_TF_losses"])
        D_cl_loss_list.append(check_point["D_epoch_cl_losses"])
        G_cl_loss_list.append(check_point["G_epoch_cl_losses"])
        real_acc_list.append(check_point["epoch_real_acc"])
        fake_acc_list.append(check_point["epoch_fake_acc"])
        FID_score.append(check_point["FID"])
        check_point["mAP_score"].to_csv(os.path.join(opts.learning_log_dir, "impression_AP_score.csv"))
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        # エポックごとに結果を表示
        LOGGER.info("所要時間 %d 分 %d 秒" % (mins, secs))
        LOGGER.info(f'\tLoss: {check_point["D_epoch_TF_losses"]:.4f}(Discriminator_TF)')
        LOGGER.info(f'\tLoss: {check_point["G_epoch_TF_losses"]:.4f}(Generator_TF)')
        LOGGER.info(f'\tLoss: {check_point["D_epoch_cl_losses"]:.4f}(Discriminator_class)')
        LOGGER.info(f'\tLoss: {check_point["G_epoch_cl_losses"]:.4f}(Generator_class)')
        LOGGER.info(f'\tacc: {check_point["epoch_real_acc"]:.4f}(real_acc)')
        LOGGER.info(f'\tacc: {check_point["epoch_fake_acc"]:.4f}(fake_acc)')
        LOGGER.info(f'\tFID: {check_point["FID"]:.4f}(FID)')
       # モデル保存のためのcheckpointファイルを作成
        if iter_start >= opts.res_step*6:
            break

    writer.close()



if __name__=="__main__":
    parser = get_parser()
    opts = parser.parse_args()
    # 再現性確保のためseed値固定
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # make dirs
    opts.log_dir, opts.weight_dir, opts.logs_GAN, opts.learning_log_dir = \
        make_logdir(os.path.join(opts.root, opts.dt_now))
    # 回すモデルの選定
    LOGGER = init_logger(opts.log_dir)
    LOGGER.info(f"================hyper parameter================ \n"
                f"device::{opts.device}\n"
                f"batch_size:{opts.batch_size}\n"
                f"g_lr:{opts.g_lr}\n"
                f"d_lr:{opts.d_lr}\n"
                f"start_iteration:{opts.start_iterations}\n"
                f"img_size:{opts.img_size}\n"
                f"w2v_dimension:{opts.w2v_dimension}\n"
                f"latent_size:{opts.latent_size}\n"
                f"num_epochs:{opts.num_epochs}\n"
                f"char_num:{opts.char_num}\n"
                f"num_critic:{opts.num_critic}\n"
                f"lambda_gp:{opts.lambda_gp}\n"
                f"lambda_class:{opts.lambda_class}\n")
    model_run(opts)
