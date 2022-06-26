__author__ = "Minghao Gou"
__version__ = "1.0"

import torch
import os
import numpy as np


def eval_once(
    writer,
    args,
    weights_folder,
    num_iter,
    net,
    data_loader,
    best_precision_0_5,
    best_precision_0_8,
    split,
    cal_accuracy_0_5,
    cal_accuracy_0_8,
    total_batch_num=50,
):
    torch.cuda.empty_cache()
    net.eval()
    with torch.no_grad():
        acc_list_0_5 = []
        acc_list_0_8 = []
        if args.local_rank == 0:
            print(f"Calculating {split} Accuracy Before Iteration {num_iter}")
        for batch_num, batch_data in enumerate(data_loader):
            if batch_num == total_batch_num:
                break
            rgb, _, label, normal = batch_data
            rgb = rgb.cuda(non_blocking=True)
            normal = normal.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            prob_map = net(rgb, normal)
            (
                acc_true_0_5,
                acc_false_0_5,
                precision_0_5,
                pred_true_num_0_5,
                topk_acc_0_5,
            ) = cal_accuracy_0_5(prob_map, label)
            acc_list_0_5.append(
                [
                    acc_true_0_5.item(),
                    acc_false_0_5.item(),
                    precision_0_5.item(),
                    pred_true_num_0_5.item(),
                    topk_acc_0_5.item(),
                ]
            )
            (
                acc_true_0_8,
                acc_false_0_8,
                precision_0_8,
                pred_true_num_0_8,
                topk_acc_0_8,
            ) = cal_accuracy_0_8(prob_map, label)
            acc_list_0_8.append(
                [
                    acc_true_0_8.item(),
                    acc_false_0_8.item(),
                    precision_0_8.item(),
                    pred_true_num_0_8.item(),
                    topk_acc_0_8.item(),
                ]
            )

        mean_acc_0_5 = np.mean(np.array(acc_list_0_5), axis=0)
        (
            mean_acc_true_0_5,
            mean_acc_false_0_5,
            mean_precision_0_5,
            mean_pred_true_num_0_5,
            mean_topk_acc_0_5,
        ) = mean_acc_0_5
        mean_acc_0_8 = np.mean(np.array(acc_list_0_8), axis=0)
        (
            mean_acc_true_0_8,
            mean_acc_false_0_8,
            mean_precision_0_8,
            mean_pred_true_num_0_8,
            mean_topk_acc_0_8,
        ) = mean_acc_0_8
        if args.local_rank == 0:
            if best_precision_0_8 < mean_precision_0_8:
                if split == "test":
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
                            "rgbd_{}_best_0_8_test.pth".format(
                                args.cfg.replace(".yaml", "")
                            ),
                        ),
                    )
                print(f"### New Best Result On {split} 0.8 ###")
                best_precision_0_8 = mean_precision_0_8
            if best_precision_0_5 < mean_precision_0_5:
                print(f"### New Best Result On {split} 0.5 ###")
                best_precision_0_5 = mean_precision_0_5
            if mean_topk_acc_0_8 > 0.3:
                if split == "test":
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
                            "rgbd_{}_topK_0_8_precision_{}_iter_{}_test.pth".format(
                                args.cfg.replace(".yaml", ""),
                                mean_topk_acc_0_8,
                                num_iter,
                            ),
                        ),
                    )

            print(
                f"---------------------\nTest mean acc true@_0_5:{mean_acc_true_0_5}, mean acc false@_0_5:{mean_acc_false_0_5}, precision@_0_5/best:{mean_precision_0_5}/{best_precision_0_5}, mean_pred_true_num@_0_5:{mean_pred_true_num_0_5}, mean_topk_acc_0_5:{mean_topk_acc_0_5}\n---------------------"
            )
            writer.add_scalar(f"acc_{split}/acc_true_0_5", mean_acc_true_0_5, num_iter)
            writer.add_scalar(
                f"acc_{split}/acc_false_0_5", mean_acc_false_0_5, num_iter
            )
            writer.add_scalar(
                f"acc_{split}/precision_0_5", mean_precision_0_5, num_iter
            )
            writer.add_scalar(
                f"acc_{split}/pred_true_num_0_5", mean_pred_true_num_0_5, num_iter
            )
            writer.add_scalar(f"acc_{split}/topk_acc_0_5", mean_topk_acc_0_5, num_iter)

            print(
                f"---------------------\nTest mean acc true@_0_8:{mean_acc_true_0_8}, mean acc false@_0_8:{mean_acc_false_0_8}, precision@_0_8/best:{mean_precision_0_8}/{best_precision_0_8}, mean_pred_true_num@_0_8:{mean_pred_true_num_0_8}, mean_topk_acc_0_8:{mean_topk_acc_0_8}\n---------------------"
            )
            writer.add_scalar(f"acc_{split}/acc_true_0_8", mean_acc_true_0_8, num_iter)
            writer.add_scalar(
                f"acc_{split}/acc_false_0_8", mean_acc_false_0_8, num_iter
            )
            writer.add_scalar(
                f"acc_{split}/precision_0_8", mean_precision_0_8, num_iter
            )
            writer.add_scalar(
                f"acc_{split}/pred_true_num_0_8", mean_pred_true_num_0_8, num_iter
            )
            writer.add_scalar(f"acc_{split}/topk_acc_0_8", mean_topk_acc_0_8, num_iter)
    torch.cuda.empty_cache()
    return best_precision_0_5, best_precision_0_8


def eval_once_depth(
    writer,
    args,
    weights_folder,
    num_iter,
    net,
    data_loader,
    best_precision_0_5,
    best_precision_0_8,
    split,
    cal_accuracy_0_5,
    cal_accuracy_0_8,
    total_batch_num=50,
):
    torch.cuda.empty_cache()
    net.eval()
    with torch.no_grad():
        acc_list_0_5 = []
        acc_list_0_8 = []
        if args.local_rank == 0:
            print(f"Calculating {split} Accuracy Before Iteration {num_iter}")
        for batch_num, batch_data in enumerate(data_loader):
            if batch_num == total_batch_num:
                break
            rgb, depth, label, normal = batch_data

            depth = depth.squeeze()  # (1,288,384)
            depth = depth.reshape((depth.shape[0], -1))  # (1,288*384)
            max_depth = depth.max(1)[0]
            min_depth = depth.min(1)[0]
            delt_depth = max_depth - min_depth
            depth = ((depth.T - min_depth) / delt_depth * 255.0).T
            depth = depth.reshape((1, 288, 384))
            depth = depth.unsqueeze(1)
            depth = torch.cat((depth, depth, depth), 1)

            depth = depth.cuda(non_blocking=True)
            prob_map = net(depth, depth)
            (
                acc_true_0_5,
                acc_false_0_5,
                precision_0_5,
                pred_true_num_0_5,
                topk_acc_0_5,
            ) = cal_accuracy_0_5(prob_map, label)
            acc_list_0_5.append(
                [
                    acc_true_0_5.item(),
                    acc_false_0_5.item(),
                    precision_0_5.item(),
                    pred_true_num_0_5.item(),
                    topk_acc_0_5.item(),
                ]
            )
            (
                acc_true_0_8,
                acc_false_0_8,
                precision_0_8,
                pred_true_num_0_8,
                topk_acc_0_8,
            ) = cal_accuracy_0_8(prob_map, label)
            acc_list_0_8.append(
                [
                    acc_true_0_8.item(),
                    acc_false_0_8.item(),
                    precision_0_8.item(),
                    pred_true_num_0_8.item(),
                    topk_acc_0_8.item(),
                ]
            )

        mean_acc_0_5 = np.mean(np.array(acc_list_0_5), axis=0)
        (
            mean_acc_true_0_5,
            mean_acc_false_0_5,
            mean_precision_0_5,
            mean_pred_true_num_0_5,
            mean_topk_acc_0_5,
        ) = mean_acc_0_5
        mean_acc_0_8 = np.mean(np.array(acc_list_0_8), axis=0)
        (
            mean_acc_true_0_8,
            mean_acc_false_0_8,
            mean_precision_0_8,
            mean_pred_true_num_0_8,
            mean_topk_acc_0_8,
        ) = mean_acc_0_8
        if args.local_rank == 0:
            if best_precision_0_8 < mean_precision_0_8:
                if split == "test":
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
                            "rgbd_{}_best_0_8_test.pth".format(
                                args.cfg.replace(".yaml", "")
                            ),
                        ),
                    )
                print(f"### New Best Result On {split} 0.8 ###")
                best_precision_0_8 = mean_precision_0_8
            if best_precision_0_5 < mean_precision_0_5:
                print(f"### New Best Result On {split} 0.5 ###")
                best_precision_0_5 = mean_precision_0_5
            if mean_topk_acc_0_8 > 0.3:
                if split == "test":
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
                            "rgbd_{}_topK_0_8_precision_{}_iter_{}_test.pth".format(
                                args.cfg.replace(".yaml", ""),
                                mean_topk_acc_0_8,
                                num_iter,
                            ),
                        ),
                    )

            print(
                f"---------------------\nTest mean acc true@_0_5:{mean_acc_true_0_5}, mean acc false@_0_5:{mean_acc_false_0_5}, precision@_0_5/best:{mean_precision_0_5}/{best_precision_0_5}, mean_pred_true_num@_0_5:{mean_pred_true_num_0_5}, mean_topk_acc_0_5:{mean_topk_acc_0_5}\n---------------------"
            )
            writer.add_scalar(f"acc_{split}/acc_true_0_5", mean_acc_true_0_5, num_iter)
            writer.add_scalar(
                f"acc_{split}/acc_false_0_5", mean_acc_false_0_5, num_iter
            )
            writer.add_scalar(
                f"acc_{split}/precision_0_5", mean_precision_0_5, num_iter
            )
            writer.add_scalar(
                f"acc_{split}/pred_true_num_0_5", mean_pred_true_num_0_5, num_iter
            )
            writer.add_scalar(f"acc_{split}/topk_acc_0_5", mean_topk_acc_0_5, num_iter)

            print(
                f"---------------------\nTest mean acc true@_0_8:{mean_acc_true_0_8}, mean acc false@_0_8:{mean_acc_false_0_8}, precision@_0_8/best:{mean_precision_0_8}/{best_precision_0_8}, mean_pred_true_num@_0_8:{mean_pred_true_num_0_8}, mean_topk_acc_0_8:{mean_topk_acc_0_8}\n---------------------"
            )
            writer.add_scalar(f"acc_{split}/acc_true_0_8", mean_acc_true_0_8, num_iter)
            writer.add_scalar(
                f"acc_{split}/acc_false_0_8", mean_acc_false_0_8, num_iter
            )
            writer.add_scalar(
                f"acc_{split}/precision_0_8", mean_precision_0_8, num_iter
            )
            writer.add_scalar(
                f"acc_{split}/pred_true_num_0_8", mean_pred_true_num_0_8, num_iter
            )
            writer.add_scalar(f"acc_{split}/topk_acc_0_8", mean_topk_acc_0_8, num_iter)
    torch.cuda.empty_cache()
    return best_precision_0_5, best_precision_0_8
