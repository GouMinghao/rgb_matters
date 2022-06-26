__author__ = "Minghao Gou"
__version__ = "1.0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from rgbd_graspnet.data import GraspNetDataset
from rgbd_graspnet.net.acc import Acc_v2
from rgbd_graspnet.net.rgb_normal_net import RGBNormalNet
from rgbd_graspnet.constant import GRASPNET_ROOT, LABEL_DIR
from rgbd_graspnet.net.eval import eval_once

import time
import argparse
import os
import yaml


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="v2.yaml", help="Yaml config file name")
parser.add_argument(
    "--cuda", default=True, type=str2bool, help="Use CUDA to train model"
)
parser.add_argument(
    "--num_workers", default=12, type=int, help="Number of workers used in data loading"
)
parser.add_argument(
    "--dataset_root", default=GRASPNET_ROOT, help="Dataset root directory path"
)
parser.add_argument("--save_folder", default="weights", help="Folder to save models")
parser.add_argument(
    "--kinect_label_root", default=LABEL_DIR, help="Root folder of kinect labels"
)
parser.add_argument(
    "--realsense_label_root",
    default=LABEL_DIR,
    help="Root folder of realsense labels",
)
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    help="Checkpoint state_dict file to resume training from",
)
parser.add_argument(
    "--tb_log_dir", default="tb_log", help="Folder to save tensorboard logs"
)
parser.add_argument(
    "--basenet_type", default="resnet", help="Type of base network to use"
)
parser.add_argument(
    "--local_rank", default=0, type=int, help="node rank for distributed training"
)

args = parser.parse_args()
if not args.cuda:
    raise ValueError("\033[031mCUDA must be used now\033[0m")

with open(os.path.join("config", args.cfg)) as yaml_file:
    train_config = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
if args.local_rank == 0:
    print("\033[034mconfig:{}\033[0m".format(train_config))

batch_size = train_config["batch_size"]
eval_batch_size = train_config["eval_batch_size"]
eval_test_batch_num = train_config["eval_train_batch_num"]
eval_train_batch_num = train_config["eval_train_batch_num"]
train_camera = train_config["train_camera"]
test_camera = train_config["test_camera"]
num_layers = train_config["num_layers"]
lr = float(train_config["lr"])
iters = train_config["iters"]
use_normal = train_config["use_normal"]
normal_only = train_config["normal_only"]
lr_decay = train_config["lr_decay"]
grad_clip = train_config["grad_clip"]

grayscale = train_config["augmentation"]["grayscale"]
colorjitter_scale = train_config["augmentation"]["colorjitter_scale"]
random_crop = train_config["augmentation"]["random_crop"]
pos_weight = train_config["loss"]["pos"]
neg_weight = train_config["loss"]["neg"]
test_split = train_config["split"]["test_split"]
train_split = train_config["split"]["train_split"]

if args.local_rank == 0:
    print("\033[034mtraining args:{}\033[0m".format(args))
if torch.cuda.device_count() > 1:
    dist.init_process_group(backend="nccl")
torch.cuda.set_device(args.local_rank)

if train_camera == "realsense":
    train_label_root = args.realsense_label_root
elif train_camera == "kinect":
    train_label_root = args.kinect_label_root
else:
    raise ValueError('camera must be "realsense" or "kinect"')

if test_camera == "realsense":
    test_label_root = args.realsense_label_root
elif test_camera == "kinect":
    test_label_root = args.kinect_label_root
else:
    raise ValueError('camera must be "realsense" or "kinect"')

net = RGBNormalNet(
    num_layers=num_layers, use_normal=use_normal, normal_only=normal_only
)
net.train()

if args.cuda:
    net = net.cuda()

if args.resume is not None:
    state_dict = torch.load(os.path.join(args.save_folder, args.resume))
    start_iter = state_dict["total_iter"]
    net.load_state_dict(state_dict["net"])
    print(f"\033[034mresumed {os.path.join(args.save_folder, args.resume)}\033[0m")

else:
    start_iter = 0

if torch.cuda.device_count() > 1:
    if args.local_rank == 0:
        print("## Using Multi GPU, Number: {} ##".format(torch.cuda.device_count()))
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], find_unused_parameters=True
    )

graspnet_test = GraspNetDataset(
    graspnet_root=args.dataset_root,
    label_root=test_label_root,
    use_normal=use_normal,
    camera=test_camera,
    split=test_split,
    grayscale=grayscale,
    colorjitter_scale=0,
    random_crop=0,
)

graspnet_train = GraspNetDataset(
    graspnet_root=args.dataset_root,
    label_root=train_label_root,
    use_normal=use_normal,
    camera=train_camera,
    split=train_split,
    grayscale=grayscale,
    colorjitter_scale=colorjitter_scale,
    random_crop=random_crop,
)
if torch.cuda.device_count() > 1:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=graspnet_train, shuffle=False
    )
    eval_test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=graspnet_test, shuffle=True
    )
    eval_train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=graspnet_train, shuffle=True
    )
if args.local_rank == 0:
    print("\033[034mdataset loaded\033[0m")
if torch.cuda.device_count() == 1:
    eval_test_dataloader = DataLoader(
        graspnet_test,
        shuffle=True,
        batch_size=eval_batch_size,
        num_workers=args.num_workers,
    )
    eval_train_dataloader = DataLoader(
        graspnet_train,
        shuffle=False,
        batch_size=eval_batch_size,
        num_workers=args.num_workers,
    )
    dataloader = DataLoader(
        graspnet_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
elif torch.cuda.device_count() > 1:
    eval_test_dataloader = DataLoader(
        graspnet_test,
        batch_size=eval_batch_size,
        sampler=eval_test_sampler,
        num_workers=args.num_workers,
    )
    eval_train_dataloader = DataLoader(
        graspnet_train,
        batch_size=eval_batch_size,
        sampler=eval_train_sampler,
        num_workers=args.num_workers,
    )
    dataloader = DataLoader(
        graspnet_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
else:
    raise ValueError("CPU Training is not supported")

if args.local_rank == 0:
    print("\033[034mdataloader loaded\033[0m")

cal_accuracy_0_5 = Acc_v2(thresh=0.5)
cal_accuracy_0_5 = cal_accuracy_0_5.cuda()

cal_accuracy_0_8 = Acc_v2(thresh=0.8)
cal_accuracy_0_8 = cal_accuracy_0_8.cuda()

# criterion = RGBD_Graspnet_Loss(
criterion = nn.MSELoss()
criterion = criterion.cuda()

optimizer = optim.Adam(net.parameters(), lr=lr)


time_str = time.ctime()
time_str = time_str.replace(":", "-")
time_str = time_str.replace(" ", "-")
if args.local_rank == 0:
    writer = SummaryWriter(args.tb_log_dir + "_" + time_str)
else:
    writer = None

weights_folder = os.path.join(
    args.save_folder, f'{args.cfg.replace(".yaml","")}_{time_str}'
)
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)

num_iter = start_iter
end_iter = start_iter + iters

best_test_precision_0_8 = 0.0
best_test_precision_0_5 = 0.0
best_train_precision_0_8 = 0.0
best_train_precision_0_5 = 0.0

while num_iter <= end_iter:
    for batch_num, batch_data in enumerate(dataloader):
        if num_iter % 400 == 0 and not num_iter == 0:
            best_test_precision_0_5, best_test_precision_0_8 = eval_once(
                writer,
                args,
                weights_folder,
                num_iter,
                net,
                eval_test_dataloader,
                best_test_precision_0_5,
                best_test_precision_0_8,
                "test",
                cal_accuracy_0_5,
                cal_accuracy_0_8,
                total_batch_num=eval_test_batch_num,
            )
            best_train_precision_0_5, best_train_precision_0_8 = eval_once(
                writer,
                args,
                weights_folder,
                num_iter,
                net,
                eval_train_dataloader,
                best_train_precision_0_5,
                best_train_precision_0_8,
                "train",
                cal_accuracy_0_5,
                cal_accuracy_0_8,
                total_batch_num=eval_train_batch_num,
            )
        if num_iter % 20000 == 0 and num_iter > 100:
            lr *= lr_decay
            optimizer = optim.Adam(net.parameters(), lr=lr)

        # load data
        net.train()
        rgb, _, label, normal = batch_data
        rgb = rgb.cuda(non_blocking=True)
        normal = normal.cuda(non_blocking=True)

        label = label.cuda(non_blocking=True)

        optimizer.zero_grad()
        prob_map = net(rgb, normal)

        loss = criterion(prob_map, label)

        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        if batch_num % 10 == 0:
            if args.local_rank == 0:
                print(f"iter:{num_iter}, batch:{batch_num}, loss:{loss}")
                writer.add_scalar("loss_train/loss", loss, num_iter)

        num_iter += 1
        if num_iter % 100 == 0:
            if args.local_rank == 0:
                state_dict = dict()
                state_dict["total_iter"] = num_iter
                if torch.cuda.device_count() > 1:
                    state_dict["net"] = net.module.state_dict()
                else:
                    state_dict["net"] = net.state_dict()
                torch.save(
                    state_dict,
                    os.path.join(
                        weights_folder,
                        "rgbd_{}_iter_{}.pth".format(args.basenet_type, num_iter),
                    ),
                )

        if num_iter == end_iter:
            break
    torch.cuda.empty_cache()

if args.local_rank == 0:
    state_dict = dict()
    state_dict["total_iter"] = num_iter
    if torch.cuda.device_count() > 1:
        state_dict["net"] = net.module.state_dict()
    else:
        state_dict["net"] = net.state_dict()
    torch.save(
        state_dict,
        os.path.join(
            weights_folder, "rgbd_{}_iter_{}.pth".format(args.basenet_type, num_iter)
        ),
    )

eval_once(
    writer,
    args,
    weights_folder,
    num_iter,
    net,
    eval_test_dataloader,
    best_test_precision_0_5,
    best_test_precision_0_8,
    "test",
    cal_accuracy_0_5,
    cal_accuracy_0_8,
    total_batch_num=50,
)
eval_once(
    writer,
    args,
    weights_folder,
    num_iter,
    net,
    eval_train_dataloader,
    best_train_precision_0_5,
    best_train_precision_0_8,
    "train",
    cal_accuracy_0_5,
    cal_accuracy_0_8,
    total_batch_num=50,
)
