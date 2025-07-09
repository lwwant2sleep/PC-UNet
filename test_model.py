

import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
import pickle

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import csv

from nets.ACC_UNet_2_11 import ACC_UNet
from nets.UCTransNet import UCTransNet
from nets.UNet_base import UNet_base
from nets.SMESwinUnet import SMESwinUnet
from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet
from nets.UNet_pp import UNetPP
import json
from utils import *
from medpy.metric.binary import hd95
import numpy as np
from sklearn.metrics import confusion_matrix
from skimage.segmentation import find_boundaries
from nets.Efficientunet.efficientunet import *

def calculate_boundary_f1(mask_true, mask_pred, mode='inner', connectivity=1):

    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)


    true_boundary = find_boundaries(mask_true, mode=mode, connectivity=connectivity)
    pred_boundary = find_boundaries(mask_pred, mode=mode, connectivity=connectivity)


    tp = np.sum(np.logical_and(true_boundary, pred_boundary))
    fp = np.sum(np.logical_and(pred_boundary, ~true_boundary))
    fn = np.sum(np.logical_and(true_boundary, ~pred_boundary))


    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1

def calculate_metrics(np_prediction, np_target):

    y_pre = np.where(np_prediction >= 0.5, 1, 0)

    y_true = np.where(np_target >= 0.5, 1, 0)

    Boundary_F1 = calculate_boundary_f1(mask_true=y_true, mask_pred=y_pre)

    confusion = confusion_matrix(y_true, y_pre)

    if confusion.shape == (1, 1):

        TN, FP, FN, TP = 0, 0, 0, confusion[0, 0] if y_true[0] == 1 else confusion[0, 0]
    else:

        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0

    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0

    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0

    Dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0

    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    return accuracy, sensitivity, specificity, Dice, miou, confusion, Boundary_F1,# img_hd95


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))

    return dice_pred, iou_pred


def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,
                                                  save_path=vis_save_path + '_predict' + model_type + '.jpg')


    input_img.to('cpu')

    input_img = input_img[0].transpose(0, -1).cpu().detach().numpy()
    labs = labs[0]


    out = output.squeeze(1).cpu().detach().numpy()

    output = output[0, 0, :, :].cpu().detach().numpy()

    if np.all(labs == 0):
        print("Warning: Ground Truth is all zeros (no foreground)!")
        img_hd95 = 316.7838379716
    elif np.all(predict_save == 0):
        print("Warning: predict is all zeros (no foreground)!")
        img_hd95 = 316.7838379716
    else:
        img_hd95 = hd95(predict_save, labs)

    if (False):
        pickle.dump({
            'input': input_img,
            'output': (output >= 0.5) * 1.0,
            'ground_truth': labs,
            'dice': dice_pred_tmp,
            'iou': iou_tmp
        },
            open(vis_save_path + '.p', 'wb'))

    if (True):
        plt.figure(figsize=(10, 3.3))
        plt.subplot(1, 3, 1)
        plt.imshow(input_img)
        plt.subplot(1, 3, 2)
        plt.imshow(labs, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow((output >= 0.5) * 1.0, cmap='gray')
        plt.suptitle(f'Dice score : {dice_pred_tmp:.4f} \nIoU : {iou_tmp:.4f} \nhd95 : {img_hd95:.4f}')
        #plt.suptitle(f'Dice score : {np.round(dice_pred_tmp, 3)}\nIoU : {np.round(iou_tmp, 3)}' )
        plt.tight_layout()
        plt.savefig(vis_save_path)
        plt.close()

    return dice_pred_tmp, iou_tmp, out, img_hd95


if __name__ == '__main__':


    test_session = config.test_session
    if config.task_name == "Glas":
        test_num = 80
        model_type = config.model_name
        model_path = "./Glas/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "BUSI":
        test_num = 120
        model_type = config.model_name
        model_path = "./BUSI/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "ISIC2018":
        test_num = 1000
        model_type = config.model_name
        model_path = "./ISIC2018/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Kvasir-SEG":
        test_num = 200
        model_type = config.model_name
        model_path = "./Kvasir-SEG/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Clinic_exp1":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp1/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"


    save_path = config.task_name + '/' + config.model_name + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    vis_path = save_path + 'visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    fp = open(save_path + 'test.result', 'a')
    fp.write(str(datetime.now()) + '\n')

    if model_type == 'ACC_UNet':
        config_vit = config.get_CTranS_config()
        model = ACC_UNet(n_channels=config.n_channels, n_classes=config.n_labels, n_filts=config.n_filts)

    elif model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNet_base':
        config_vit = config.get_CTranS_config()
        model = UNet_base(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNetPP':
        config_vit = config.get_CTranS_config()
        model = UNetPP(in_channel=config.n_channels,out_channel=config.n_labels)
    elif model_type == 'efficientunet':
        model = get_efficientunet_b3(out_channels=config.n_labels, pretrained=False)
    elif model_type == 'SwinUnet':
        model = SwinUnet()
        #model.load_from()

    elif model_type == 'SMESwinUnet':
        model = SMESwinUnet(n_channels=config.n_channels, n_classes=config.n_labels)
        #model.load_from()

    elif model_type.split('_')[0] == 'MultiResUnet1':
        model = MultiResUnet(n_channels=config.n_channels, n_classes=config.n_labels,
                             nfilt=int(model_type.split('_')[1]), alpha=float(model_type.split('_')[2]))


    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    dice_pred = 0.0
    iou_pred = 0.0
    hd95_pred=0.0
    dice_ens = 0.0

    predictions = []

    targets = []


    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:

        csv_path = vis_path + f'{model_type}_{config.task_name}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Dice', 'HD95'])

        for i, (sampled_batch, names) in enumerate(test_loader, 1):


            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())


            targets.append(test_label.squeeze(1).cpu().detach().numpy())

            lab = test_label.data.numpy()

            height, width = config.img_size, config.img_size
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t,out ,HD95= vis_and_save_heatmap(model, input_img, None, lab,
                                                           vis_path + str(i) + '.png',
                                                           dice_pred=dice_pred, dice_ens=dice_ens)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'{i}.png', dice_pred_t, HD95])


            predictions.append(out)
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            hd95_pred+=HD95
            torch.cuda.empty_cache()
            pbar.update()


    predictions = np.array(predictions).reshape(-1)

    targets = np.array(targets).reshape(-1)
    accuracy, sensitivity, specificity, Dice, miou, confusion, Boundary_F1= calculate_metrics(predictions, targets)
    print(f"dice_pred {(dice_pred / test_num):.4f}")
    print(f"iou_pred {(iou_pred / test_num):.4f}")
    print(f"hd95 {(hd95_pred / test_num):.2f}")

    print(f"sensitivity {sensitivity:.4f}")
    print(f"specificity {specificity:.4f}")


    fp.write(f"dice_pred : {dice_pred / test_num}\n")
    fp.write(f"iou_pred : {iou_pred / test_num}\n")
    fp.write(f"hd95 : {hd95_pred / test_num}\n")
    fp.write(f"accuracy : {accuracy}\n")
    fp.write(f"sensitivity : {sensitivity}\n")
    fp.write(f"specificity : {specificity}\n")

    fp.close()

