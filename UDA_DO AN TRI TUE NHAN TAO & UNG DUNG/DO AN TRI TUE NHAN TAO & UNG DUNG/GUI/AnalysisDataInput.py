"""
Author: Ngo Van Uc
Gmail: ngovanuc.1508@gmail.com
License: (contact to me)
"""
# INPUTS: FILES, FOLDERS, NUMBER OF FILES AND FOLDERS

import os
from os import listdir
os.environ['JAX_PLATFORMS'] = ''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import glob
import time
import copy
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from cv2 import imread
from cv2 import imshow
from cv2 import resize
from PIL import Image
from skimage.io import imread
from tqdm import tqdm_notebook as tqdm
from keras.models import load_model


global folders
global files
global num_folders
global num_files


path = 'E:/UDA_LEARNING/UDA_DO AN TRI TUE NHAN TAO & UNG DUNG/BREAST CANCER/data/IDC_regular_ps50_idx5/9173'
save_img_at = 'E:/UDA_LEARNING/UDA_DO AN TRI TUE NHAN TAO & UNG DUNG/DO AN TRI TUE NHAN TAO & UNG DUNG/img'
characters = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))


def path_to_list_files(path):
    files = glob.glob(path + '/*/*')
    return files


def paths_to_dataframe(files, num_files):
    data = pd.DataFrame(index=np.arange(0, num_files), columns=['path', 'target'])
    k = 0
    for f in files:
        data.loc[k, 'path'] = f
        if re.findall(r'class0', f):
            data.loc[k, 'target'] = 0
        elif re.findall(r'class1', f):
            data.loc[k, 'target'] = 1
        k += 1
    
    return data


def data_to_Pos_Neg(data):
    # df1 = df.iloc[:half_len].reset_index(drop=True)
    # df2 = df.iloc[half_len:].reset_index(drop=True)
    Pos_path = pd.DataFrame(data.groupby("target").get_group(0)).reset_index(drop=True)
    Neg_path = pd.DataFrame(data.groupby('target').get_group(1)).reset_index(drop=True)
    return Pos_path, Neg_path


def fig_class(data):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    class_counts = data['target'].value_counts()
    class_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Số lượng ảnh của mỗi class')

    random_characters = np.random.choice(characters, size=10)
    random_characters = ''.join(random_characters)
    full_name = 'SoLuongAnhCuaMoiClass_' + random_characters + '.png'
    img_path = save_img_at + '/' + full_name
    fig.savefig(img_path)
    return img_path


def some_of_Neg(neg_paths):
    # hàm này không nên dùng
    fig, ax = plt.subplots(5, 10, figsize=(20,10))
    for n in range(5):
        for m in range(10):
            image = imread(neg_paths.loc[np.random.randint(0, len(neg_paths)), 'path'])
            ax[n, m].imshow(image)
            ax[n, m].grid(False)
    
    random_characters = np.random.choice(characters, size=10)
    random_characters = ''.join(random_characters)
    full_name = 'MauDuongTinh_' + random_characters + '.png'
    img_path = save_img_at + '/' + full_name
    fig.savefig(img_path)
    return img_path


def some_of_Peg(pos_paths):
    # hàm này không nên dùng
    fig, ax = plt.subplots(5, 10, figsize=(20,10))
    for n in range(5):
        for m in range(10):
            image = imread(pos_paths.loc[np.random.randint(0, len(pos_paths)), 'path'])
            ax[n, m].imshow(image)
            ax[n, m].grid(False)
    
    random_characters = np.random.choice(characters, size=10)
    random_characters = ''.join(random_characters)
    full_name = 'MauAmTinh_' + random_characters + '.png'
    img_path = save_img_at + '/' + full_name
    fig.savefig(img_path)
    return img_path


# hiển thị hình ảnh biểu tượng nhị phân trên hệ trục tọa độ - coordinate
# 8863_idx5_x51_y1251_class0
# u_xX_yY_classC.png

def path_to_coordinate_frame(df_paths, target):
    data = pd.DataFrame(index=np.arange(0, len(df_paths)), columns=['x', 'y', 'target', 'path'])
    for index in range(0, len(df_paths)):
        x = re.findall(r'_x[0-9]+', df_paths.loc[index, 'path'])
        y = re.findall(r'_y[0-9]+', df_paths.loc[index, 'path'])
        x = int(x[0][2:len(x[0])])
        y = int(y[0][2:len(y[0])])
        data.loc[index, 'x'] = x
        data.loc[index, 'y'] = y
        data.loc[index, 'target'] = target
        data.loc[index, 'path'] = df_paths.loc[index, 'path']
    return data

def total_image_to_coordinate(pos_path, neg_paths):
    pos = path_to_coordinate_frame(pos_path, 0)
    neg = path_to_coordinate_frame(neg_paths, 1)
    df_concat = pd.concat([pos, neg], axis=0)
    
    # trực quan hóa nhãn nhị phân trên mỗi lát cắt
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(df_concat.x.values, df_concat.y.values, c=df_concat.target.values, cmap='coolwarm', s=20);
    ax.set_title('Images on coordinate')
    ax.set_xlabel('x coord')
    ax.set_ylabel('y coord')

    random_characters = np.random.choice(characters, size=10)
    random_characters = ''.join(random_characters)
    full_name = 'Coordinate_' + random_characters + '.png'
    img_path = save_img_at + '/' + full_name
    fig.savefig(img_path)
    return img_path


# hiển thị toàn bộ hình ảnh trên slice thực
def visualise_breast_tissue(df_concat):
    example_df = df_concat
    max_point = [example_df.y.max()-1, example_df.x.max()-1]
    grid = 255*np.ones(shape = (max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    mask = 255*np.ones(shape = (max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    broken_patches = []
    for n in range(len(example_df)):
        try:
            image = imread(example_df.path.values[n])            
            target = example_df.target.values[n]            
            x_coord = np.int32(example_df.x.values[n])
            y_coord = np.int32(example_df.y.values[n])
            x_start = x_coord - 1
            y_start = y_coord - 1
            x_end = x_start + 50
            y_end = y_start + 50
            grid[y_start:y_end, x_start:x_end] = image
            if target == 1:
                mask[y_start:y_end, x_start:x_end, 0] = 250
                mask[y_start:y_end, x_start:x_end, 1] = 0
                mask[y_start:y_end, x_start:x_end, 2] = 0            
        except ValueError:
            broken_patches.append(example_df.path.values[n])
    
    return grid, mask, broken_patches


def tatal_image_to_slice(pos_path, neg_paths):
    pos = path_to_coordinate_frame(pos_path, 0)
    neg = path_to_coordinate_frame(neg_paths, 1)
    df_concat = pd.concat([pos, neg], axis=0)
    grid, mask, broken_patches = visualise_breast_tissue(df_concat)

    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].imshow(grid, alpha=0.9)
    ax[1].imshow(mask, alpha=0.8)
    ax[1].imshow(grid, alpha=0.7)
    ax[0].grid(False)
    ax[1].grid(False)
    for m in range(2):
        ax[m].set_xlabel("x-coord")
        ax[m].set_ylabel("y-coord")
    ax[0].set_title("Breast tissue slice of patient")
    ax[1].set_title("Cancer tissue colored red \n of patient");

    random_characters = np.random.choice(characters, size=10)
    random_characters = ''.join(random_characters)
    full_name = 'BreastTissueSlice_' + random_characters + '.png'
    img_path = save_img_at + '/' + full_name
    fig.savefig(img_path)
    return img_path



files = path_to_list_files(path)
data = paths_to_dataframe(files=files, num_files=len(files))
print(fig_class(data))
pos_paths, neg_paths = data_to_Pos_Neg(data)

print(pos_paths.head())
print(len(pos_paths))
print(neg_paths.head())
print(len(neg_paths))

print(some_of_Neg(neg_paths))
print(some_of_Peg(pos_paths))

print(pos_paths.loc[0, 'path'])
print(os.path.isfile(pos_paths.loc[0, 'path'])) # type: ignore

print(path_to_coordinate_frame(pos_paths, 0).head()) # get x, y, target, path
print(path_to_coordinate_frame(neg_paths, 1).head())

print(total_image_to_coordinate(pos_paths, neg_paths))
print(tatal_image_to_slice(pos_paths, neg_paths))