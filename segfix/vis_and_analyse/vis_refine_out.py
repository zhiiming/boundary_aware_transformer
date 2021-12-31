import os
from tqdm import tqdm
import cv2 as cv
import numpy as np
from PIL import Image
import shutil

CLASSES = ('ignore', 'Rock', 'Gravel', 'Earth', 'Packaging', 'Wood',
           'Others', 'Mixed', 'Grip', 'Truck', 'background')

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0]]


def lblsave(filename, lbl):
    import imgviz
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError("[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename)


def rebuild_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)


if __name__ == '__main__':
    project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    refine_out_path = os.path.join(project_path, 'refine_out')
    img_name_list = os.listdir(refine_out_path)
    vis_refine_out_path = os.path.join(project_path, 'visualization_out', 'vis_refine_out')
    os.makedirs(vis_refine_out_path, exist_ok=True)
    for img_name in tqdm(img_name_list):
        img_path = os.path.join(refine_out_path, img_name)
        img = cv.imread(img_path)
        out_path = os.path.join(vis_refine_out_path, img_name)
        img = img[:, :, 0]
        img = np.where(img == 255, 0, img)
        nn = np.unique(img)
        lblsave(out_path, img)
