import torch
import argparse
import os
import cv2
import numpy as np
import torch.optim as optim
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from models.network import PRIDNet
from dataset import Dataset
from torch import nn
from common import save_samples, save_checkpoint
import torch.nn.functional as F
from models.losses import MS_SSIM
from models.losses import total_variation_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default='train')
    parser.add_argument('--val-dir', type=str, default='val')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint')
    parser.add_argument('--checkpoint-epoch', type=int, default=0)
    parser.add_argument('--save-image-dir', type=str, default='results')
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--w_tv', type=float, default=0.02)

    return parser.parse_args()


def check_params(args):
    data_path = os.path.join(args.train_dir, 'origin')
    noise_path = os.path.join(args.train_dir, 'noise')
    val_path = os.path.join(args.val_dir, 'origin')
    val_noise_path = os.path.join(args.val_dir, 'noise')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'training_dataset not found {data_path}')
    if not os.path.exists(noise_path):
        raise FileNotFoundError(f'noise_dataset not found {noise_path}')
    if not os.path.exists(val_path):
        raise FileNotFoundError(f'val_dataset not found {val_path}')
    if not os.path.exists(val_noise_path):
        raise FileNotFoundError(f'val_dataset not found {val_noise_path}')

    if not os.path.exists(args.save_image_dir):
        print(f'* {args.save_image_dir} does not exist, creating...')
        os.makedirs(args.save_image_dir)

    if not os.path.exists(args.checkpoint_dir):
        print(f'* {args.checkpoint_dir} does not exist, creating...')
        os.makedirs(args.checkpoint_dir)


def main(args):
    check_params(args)

    print("Init models...")

    model = PRIDNet().cuda()

    # Create DataLoader
    data_loader = DataLoader(
        Dataset(args.train_dir),
        batch_size=args.batch_size,
        num_workers=cpu_count(),
        pin_memory=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        Dataset(args.val_dir),
        batch_size=1,
        num_workers=cpu_count(),
    )

    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    criterion_ssim = MS_SSIM(channel=1)
    criterion_tv = total_variation_loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    #####       下载权重         #####
    start_e = args.checkpoint_epoch
    if start_e != 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, '_' + str(start_e) + '.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_e = checkpoint['epoch'] + 1
        else:
            start_e = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2, last_epoch=start_e - 1)

    min_loss = float('inf')
    for e in range(start_e, args.epochs):
        model.train()
        loss_sigma = 0.0  # 记录一个epoch的loss之和
        l1_sigma = 0.0
        l2_sigma = 0.0
        ssim_sigma = 0.0
        tv_sigma = 0.0

        print(f"Epoch {e}/{args.epochs}")

        for i, (origin, noise) in enumerate(data_loader):
            # To cuda
            origin = origin.cuda()
            noise = noise.cuda()

            # ---------------- TRAIN ---------------- #
            optimizer.zero_grad()
            output = model(noise)

            loss_l1 = criterion_l1(origin, output)
            loss_l2 = criterion_l2(origin, output)
            loss_ssim = criterion_ssim(origin, output)
            loss_tv = criterion_tv(output)
            loss = loss_l1 + loss_l2 + loss_ssim + args.w_tv * loss_tv

            loss.requires_grad_(True)
            loss.backward()

            optimizer.step()

            # 统计预测信息
            loss_sigma += loss.item()
            l1_sigma += loss_l1.item()
            l2_sigma += loss_l2.item()
            ssim_sigma += loss_ssim.item()
            tv_sigma += loss_tv.item()

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % 10 == 9:
                loss_avg = loss_sigma / 10
                l1_avg = l1_sigma / 10
                l2_avg = l2_sigma / 10
                ssim_avg = ssim_sigma / 10
                tv_avg = tv_sigma / 10

                loss_sigma = 0.0
                l1_sigma = 0.0
                l2_sigma = 0.0
                ssim_sigma = 0.0
                tv_sigma = 0.0
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.5f}  L1_loss: {:.5f}  L2_loss: {:.5f}  MS-SSIM_loss: {:.5f}  tv_loss: {:.5f}".format(
                        e + 1, args.epochs, i + 1, len(data_loader), loss_avg, l1_avg, l2_avg, ssim_avg, tv_avg))

        if e % args.save_interval == 0:
            save_checkpoint(model, optimizer, e + 1, args)

            # 在验证集上观察并保存最好的模型
            print('Start validation...')
            loss_sigma = 0.0
            model.eval()
            for i, (origin, noise) in enumerate(val_loader):
                origin = origin.cuda()
                noise = noise.cuda()

                # forward
                output = model(noise)
                output.detach_()

                # 计算loss
                loss_l1 = criterion_l1(origin, output)
                loss_l2 = criterion_l2(origin, output)
                loss_ssim = criterion_ssim(origin, output)
                loss_tv = criterion_tv(output)
                loss = loss_l1 + loss_l2 + loss_ssim + args.w_tv * loss_tv
                loss_sigma += loss.item()

            loss_sigma /= len(val_loader)
            if loss_sigma < min_loss:
                min_loss = loss_sigma
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': e,
                }
                path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, path)
            print(f'epoch{e} avg_loss = {loss_sigma}, min_loss = {min_loss}')

        scheduler.step()  # 更新学习率


if __name__ == '__main__':
    args = parse_args()

    print("# ==== Train Config ==== #")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("==========================")

    main(args)

