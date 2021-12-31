import os
import shutil
from tqdm import tqdm
import cv2 as cv
import numpy as np
# import mmcv
from PIL import Image
# TODO clip the img and label respectively. generate background category and the ignore


CLASSES = ['ignore', 'Rock', 'Gravel', 'Earth', 'Packaging', 'Wood',
           'Others', 'Mixed', 'Grip', 'Truck', 'background']

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0]]


def crop_img(img):
    if len(img.shape) == 2:
        h, w = img.shape
        cropped_img = img[420:h - 420, :]
        return cropped_img
    else:
        h, w, c = img.shape
        cropped_img = img[420:h - 420, :, :]
        return cropped_img


def rebuild_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

def lblsave(filename, lbl):
    import imgviz
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename[:-3] + 'png')
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename
        )

def phrase_label_generate_background():
    github_path = os.path.join(os.getcwd(), '..')
    dataset_path = os.path.join(github_path, 'dataset/data_dataset_voc')
    output_dataset_path = os.path.join(github_path, 'dataset/dataset_ignore')
    input_img_folder_path = os.path.join(dataset_path, 'JPEGImages')
    output_img_folder_path = os.path.join(output_dataset_path, 'JPEGImages')
    input_lab_folder_path = os.path.join(dataset_path, 'SegmentationClassPNG')
    output_lab_folder_path = os.path.join(output_dataset_path, 'SegmentationClassPNG')
    rebuild_folder(output_dataset_path)
    rebuild_folder(output_img_folder_path)
    rebuild_folder(output_lab_folder_path)
    img_name_list = os.listdir(input_img_folder_path)
    img_name_list.sort()
    for img_name in tqdm(img_name_list):
        in_img_path = os.path.join(input_img_folder_path, img_name)
        out_img_path = os.path.join(output_img_folder_path, img_name)
        in_lab_path = os.path.join(input_lab_folder_path, img_name[:-3] + 'png')
        if not os.path.exists(in_lab_path):
            continue
        out_lab_path = os.path.join(output_lab_folder_path, img_name[:-3] + 'png')
        img = cv.imread(in_img_path)
        img = crop_img(img)
        lab = Image.open(in_lab_path)
        lab_np = np.array(lab)
        img_max = np.max(lab_np)
        lab_np_ori = crop_img(lab_np)
        lab_np = process_lab_single_for_PIL(lab_np_ori.copy(), img_max)
        lblsave(out_lab_path, lab_np)
        cv.imwrite(out_img_path, img)

def process_lab_single_for_PIL(lab, img_max):
    in_bg_mask = np.where(lab == 0, 1, 0)
    in_bg_mask = in_bg_mask.astype('uint8')
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(in_bg_mask, kernel, iterations=10)
    lab[erosion == 1] = img_max + 1
    return lab


if __name__ == '__main__':
    phrase_label_generate_background()