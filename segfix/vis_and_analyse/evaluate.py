import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from metrics import eval_metrics
from terminaltables import AsciiTable
import cv2 as cv

CLASSES = ['background', 'Rock', 'Gravel', 'Earth', 'Packaging', 'Wood',
           'Others', 'Mixed', 'Grip', 'Truck']

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0]]


def get_classes_and_palette(classes=None, palette=None):
    CLASSES = ['Rock', 'Gravel', 'Earth', 'Packaging', 'Wood',
               'Others', 'Mixed', 'Grip', 'Truck', 'background']

    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0]]
    classes = ['Rock', 'Gravel', 'Earth', 'Packaging', 'Wood']
    # classes = ['Rock', 'Gravel', 'Earth', 'Packaging', 'Wood', 'Others', 'Mixed', 'Grip', 'Truck', 'background']
    class_names = classes
    label_map = {}
    for i, c in enumerate(CLASSES):
        if c not in class_names:
            label_map[i] = -1
        else:
            label_map[i] = classes.index(c)

    palette = []
    for old_id, new_id in sorted(
            label_map.items(), key=lambda x: x[1]):
        if new_id != -1:
            palette.append(PALETTE[old_id])
    palette = type(PALETTE)(palette)
    return class_names, palette


def convert_to_palette(seg, output_path=None):
    palette = np.array(PALETTE)
    assert palette.shape[0] == 10
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    if output_path is not None:
        cv.imwrite(output_path, color_seg)
    else:
        cv.imwrite('output_palette.jpg', color_seg)

def replace_value(selected_classes, all_classes):
    value_map = {}
    count = 0
    for idx, class_name in enumerate(all_classes):
        if not class_name in selected_classes:
            value_map[idx] = 0
            continue
        count = count + 1
        value_map[idx] = count
    return value_map

def process_evaluate(pred_list, gt_list, selected_classes):
    all_classes = ('ignore', 'Rock', 'Gravel', 'Earth', 'Packaging', 'Wood',
                   'Others', 'Mixed', 'Grip', 'Truck', 'background')
    class_for_show = ('ignore',) + selected_classes
    num_classes = len(class_for_show)
    value_map = replace_value(selected_classes, all_classes)
    ret_metrics = eval_metrics(results=pred_list,
                               gt_seg_maps=gt_list,
                               num_classes=num_classes,
                               ignore_index=0,
                               metrics=['mIoU'],
                               nan_to_num=None,
                               label_map=dict(),
                               reduce_zero_label=False)
    metric = ['mIoU']
    class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    for i in range(num_classes):
        class_table_data.append([class_for_show[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                          ['m' + head
                           for head in class_table_data[0][1:]] + ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in ret_metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                              [ret_metrics_mean[1]] +
                              [ret_metrics_mean[0]])
    print('per class results:')
    table = AsciiTable(class_table_data)
    print('\n' + table.table)
    print('Summary:')
    table = AsciiTable(summary_table_data)

    print('\n' + table.table)

def process_npy():
    all_classes = ('ignore', 'Rock', 'Gravel', 'Earth', 'Packaging', 'Wood',
                   'Others', 'Mixed', 'Grip', 'Truck', 'background')

    selected_classes = ('Rock', 'Gravel', 'Earth', 'Packaging', 'Wood')
    class_for_show = ('ignore', ) + selected_classes
    num_classes = len(class_for_show)
    value_map = replace_value(selected_classes, all_classes)
    npy_root_path = os.path.join(os.getcwd(), '..', 'SegFormer')
    pred_folder_path = os.path.join(npy_root_path, 'saved_npy_5_categories')
    gt_folder_path = os.path.join(os.getcwd(), '..', 'dataset', 'dataset_ignore/SegmentationClassPNG')
    file_list = os.listdir(pred_folder_path)
    file_list.sort()
    pred_list = []
    gt_list = []
    count = 0
    for file_name in tqdm(file_list):
        pred_path = os.path.join(pred_folder_path, file_name)
        gt_path = os.path.join(gt_folder_path, file_name[:-3] + 'png')
        if not os.path.exists(gt_path):
            continue
        gt_png = Image.open(gt_path)
        gt = np.array(gt_png)
        new_gt = np.copy(gt)
        for k, v in value_map.items():
            new_gt[gt == k] = v
        pred_logits = np.load(pred_path)
        pred_logits = pred_logits[0]
        # shape = (10, 1080, 1920)
        pred = np.argmax(pred_logits, axis=0) + 1

        pred_list.append(pred)
        gt_list.append(new_gt)
        # count = count + 1
        # if count > 10:
        #     break
    ret_metrics = eval_metrics(results=pred_list,
                               gt_seg_maps=gt_list,
                               num_classes=num_classes,
                               ignore_index=0,
                               metrics=['mIoU'],
                               nan_to_num=None,
                               label_map=dict(),
                               reduce_zero_label=False)
    metric = ['mIoU']
    class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    for i in range(num_classes):
        class_table_data.append([class_for_show[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                          ['m' + head
                           for head in class_table_data[0][1:]] + ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in ret_metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                              [ret_metrics_mean[1]] +
                              [ret_metrics_mean[0]])
    print('per class results:')
    table = AsciiTable(class_table_data)
    print('\n' + table.table)
    print('Summary:')
    table = AsciiTable(summary_table_data)

    print('\n' + table.table)

def process():
    all_classes = ('ignore', 'Rock', 'Gravel', 'Earth', 'Packaging', 'Wood',
                   'Others', 'Mixed', 'Grip', 'Truck', 'background')

    selected_classes = ('Rock', 'Gravel', 'Earth', 'Packaging', 'Wood',
                   'Others', 'Mixed', 'Grip', 'Truck', 'background')
    class_for_show = ('ignore', ) + selected_classes
    num_classes = len(class_for_show)
    value_map = replace_value(selected_classes, all_classes)
    pred_folder_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'visualization_out/vis_refine_out'))
    gt_folder_path = os.path.abspath(os.path.join(os.getcwd(), '../../..', 'dataset', 'dataset_ignore/SegmentationClassPNG'))
    file_list = os.listdir(pred_folder_path)


    file_list.sort()
    pred_list = []
    gt_list = []
    count = 0
    for file_name in tqdm(file_list):
        pred_path = os.path.join(pred_folder_path, file_name)
        gt_path = os.path.join(gt_folder_path, file_name)
        if not os.path.exists(gt_path):
            continue
        gt_png = Image.open(gt_path)
        gt = np.array(gt_png)
        new_gt = np.copy(gt)
        for k, v in value_map.items():
            new_gt[gt == k] = v

        #####
        # rr = np.unique(gt)
        # for index in rr:
        #     pixel_num = np.sum(gt == index)
        #     print('category index: %d, number of pixel: %d' % (index, pixel_num))
        ####
        pred_png = Image.open(pred_path)
        pred = np.array(pred_png)

        pred_list.append(pred)
        gt_list.append(new_gt)
        # count = count + 1
        # if count > 10:
        #     break
    ret_metrics = eval_metrics(results=pred_list,
                               gt_seg_maps=gt_list,
                               num_classes=num_classes,
                               ignore_index=0,
                               metrics=['mIoU'],
                               nan_to_num=None,
                               label_map=dict(),
                               reduce_zero_label=False)
    metric = ['mIoU']
    class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    for i in range(num_classes):
        class_table_data.append([class_for_show[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                          ['m' + head
                           for head in class_table_data[0][1:]] + ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in ret_metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                              [ret_metrics_mean[1]] +
                              [ret_metrics_mean[0]])
    print('per class results:')
    table = AsciiTable(class_table_data)
    print('\n' + table.table)
    print('Summary:')
    table = AsciiTable(summary_table_data)

    print('\n' + table.table)

if __name__ == '__main__':
    process()
