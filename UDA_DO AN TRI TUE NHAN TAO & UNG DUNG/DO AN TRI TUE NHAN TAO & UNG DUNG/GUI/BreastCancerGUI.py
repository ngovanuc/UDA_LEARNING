"""
Author: Ngo Van Uc
Gmail: ngovanuc.1508@gmail.com
License: (contact to me)
"""
import os
os.environ['JAX_PLATFORMS'] = ''
import re
import glob
import time
import threading

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import resize
import PIL
from PIL import Image
from PIL import ImageTk
from skimage.io import imread
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import load_model
# from keras.models import load_weights


import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askdirectory

# from BreastCancerGUI_Welcome import *
# from BreastCancerGUI_Prediction import *
# from BreastCancerGUI_Whiling import *
# from BreastCancerGUI_Result import *



# Prediction Block
def on_select_gender(event):
    selected_value = gender_combo.get()
    gender = selected_value
    print("Selected value:", gender)

def on_select_typeOfBcs(event):
    selected_value = typeOfBCa_combo.get()
    ztypeOfBCa = selected_value
    print("Selected value:", typeOfBCa)

def on_select_scanned(event):
    selected_value = tyOfScanned_combo.get()
    scanned = selected_value
    print("Selected value:", scanned)

def submit_success():
    submited_window = tk.Tk()
    submited_window.title('Thông báo')
    frame = tk.Frame(master=submited_window, background='#222222', width=400, height=200, )
    frame.pack_propagate(False)
    frame.pack(fill=tk.BOTH, expand=True)
    label = tk.Label(master=frame, text="Thành công!\nĐóng cửa sổ này...", height=6, font=("Verdana", 14), fg='white', bg='#222222')
    label.pack()
    button = tk.Button(master=frame, text="Đóng cửa sổ", font=("Verdana", 10), bg='#39b70c', command=submited_window.destroy)
    button.pack()

def submit_event():
    global first_name
    global last_name
    global id_patient
    global age
    global gender
    global typeOfBCa
    global scanned

    first_name = first_name_entry.get()
    last_name = last_name_entry.get()
    id_patient = id_patient_entry.get()
    age = age_entry.get()
    gender = gender_combo.get()
    typeOfBCa = typeOfBCa_combo.get()
    scanned = tyOfScanned_combo.get()

    print(first_name)
    print(last_name)
    print(id_patient)
    print(age)
    print(gender)
    print(typeOfBCa)
    print(scanned)
    
    submit_success()
    return first_name, last_name, id_patient, age, gender, typeOfBCa, scanned

def confirm_clear():
    submited_window = tk.Tk()
    submited_window.title('Thông báo')
    frame = tk.Frame(master=submited_window, background='#222222', width=400, height=200, )
    frame.pack_propagate(False)
    frame.pack(fill=tk.BOTH, expand=True)
    label = tk.Label(master=frame, text="Đã xóa thành công!\nĐóng cửa sổ này...", height=6, font=("Verdana", 14), fg='white', bg='#222222')
    label.pack()
    button = tk.Button(master=frame, text="Đóng", font=("Verdana", 10), bg='#d11919', command=submited_window.destroy)
    button.pack()    

def delete_event():
    # global first_name
    # global last_name
    # global id_patient
    # global age
    # global gender
    # global typeOfBCa
    # global scanned
    
    first_name = first_name_entry.delete(0, tk.END)
    last_name = last_name_entry.delete(0, tk.END)
    id_patient = id_patient_entry.delete(0, tk.END)
    age = age_entry.delete(0, tk.END)
    gender = gender_combo.delete(0, tk.END)
    typeOfBCa = typeOfBCa_combo.delete(0, tk.END)
    scanned = tyOfScanned_combo.delete(0, tk.END)

    confirm_clear()

    # print(first_name)
    # print(last_name)
    # print(id_patient)
    # print(age)
    # print(gender)
    # print(typeOfBCa)
    # print(scanned)

def open_directory_dialog():
    global folder_path
    global path_label
    folder_path = askdirectory()
    if folder_path:
        print("Đã chọn thư mục tại: ", folder_path)
        path_label.config(text='' + folder_path)
    else:
        path_label.config(text="Đường dẫn thư mục không phù hợp, vui lòng chọn lại...")
        folder_path = None

def return_error():
        back_window = tk.Tk()
        back_window.title('Thông báo')
        frame = tk.Frame(master=back_window, background='#222222', width=600, height=200, )
        frame.pack_propagate(False)
        frame.pack(fill=tk.BOTH, expand=True)
        label = tk.Label(master=frame, text="[CẢNH BÁO]\nĐường dẫn không hợp lệ hoặc chưa được chọn.\nHãy chọn đường dẫn hợp lệ!", height=6, font=("Verdana", 14), fg='white', bg='#222222')
        label.pack()
        button = tk.Button(master=frame, text="Đóng", font=("Verdana", 10), bg='#d11919', command=back_window.destroy)
        button.pack()

def to_result():
    try:
        if os.path.exists(folder_path): # type: ignore
            print("Đường dẫn tồn tại: ", folder_path)
            if os.path.isdir(folder_path): # type: ignore
                print("Thư mục hợp lệ: ", folder_path)
                prediction_window.destroy()
                window_whiling()
        else:
            return_error()
    except TypeError as e:
        return_error()

def window_prediction():
    global prediction_window
    prediction_window = tk.Tk()
    prediction_window.title('Phần mềm phát hiện tế bào ung thư vú trên hình ảnh sinh thiết (Biểu mô ống xâm lấn - IDC)')
    screen_width = prediction_window.winfo_screenwidth()
    screen_height = prediction_window.winfo_screenheight()

    # CONTAINER FRAME
    container_frame = tk.Frame(master=prediction_window, bg='#323232', width=screen_width, height=screen_height)
    container_frame.pack_propagate(False)
    container_frame.pack(fill=tk.BOTH, expand=True)

    # LEFT FRAME
    left_frame = tk.Frame(master=container_frame, background='#222222', width=600, height=screen_height, border=3)
    left_frame.pack_propagate(False)
    left_frame.pack(anchor="nw", side=tk.LEFT)
    label = tk.Label(master=left_frame,
                    text="Thông tin hồ sơ bệnh nhân",
                    font=("Verdana", 15, 'bold'),
                    bg='#222222',
                    fg='white',
                    padx=60,
                    pady=60)
    label.pack()
    empty_frame = tk.Frame(master=left_frame, width= 600, height=1, bg='yellow')
    empty_frame.pack()

    # FORM FRAME
    form_frame = tk.Frame(master=left_frame, width= 600, height=400, bg='#222222')
    form_frame.pack()

    global first_name_entry
    first_name_label = tk.Label(master=form_frame, text="First Name:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    first_name_entry = tk.Entry(master=form_frame, width=30, font=("Verdana", 14))
    first_name_label.grid(row=0, column=0, sticky="e", padx=10, pady=10)
    first_name_entry.grid(row=0, column=1, padx=15, pady=10)

    global last_name_entry
    last_name_label = tk.Label(master=form_frame, text="Last Name:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    last_name_entry = tk.Entry(master=form_frame, width=30, font=("Verdana", 14))
    last_name_label.grid(row=1, column=0, sticky="e", padx=10, pady=10)
    last_name_entry.grid(row=1, column=1, padx=15, pady=10)

    global id_patient_entry
    id_patient_label = tk.Label(master=form_frame, text="Patient ID:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    id_patient_entry = tk.Entry(master=form_frame, width=30, font=("Verdana", 14))
    id_patient_label.grid(row=2, column=0, sticky="e", padx=10, pady=10)
    id_patient_entry.grid(row=2, column=1, padx=15, pady=10)

    global age_entry
    age_label = tk.Label(master=form_frame, text="Age:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    age_entry = tk.Entry(master=form_frame, width=30, font=("Verdana", 14))
    age_label.grid(row=3, column=0, sticky="e", padx=10, pady=10)
    age_entry.grid(row=3, column=1, padx=15, pady=10)

    global gender_selected
    global gender_combo
    gender = ["Male", "Female"]
    gender_label = tk.Label(master=form_frame, text="Gender:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    gender_combo = ttk.Combobox(master=form_frame, values=gender, font=("Verdana", 14), width=7)
    gender_label.grid(row=4, column=0, sticky="e", padx=10, pady=10)
    gender_combo.grid(row=4, column=1)
    gender_combo.bind("<<ComboboxSelected>>", on_select_gender)

    global typeOfBCa_selected
    global typeOfBCa_combo
    typeOfBCa = ["Invasive ductal carcinoma - IDC",
                "Invasive lobular carcinoma - ILC",
                "Inflammatory breast cancer - IBC",
                "Triple Negative Breast Cancer - TNBC",
                "Other"]
    typeOfBCa_label = tk.Label(master=form_frame, text="Type of BCa:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    typeOfBCa_combo = ttk.Combobox(master=form_frame, values=typeOfBCa, font=("Verdana", 14), width=28)
    typeOfBCa_label.grid(row=5, column=0, sticky="e", padx=10, pady=10)
    typeOfBCa_combo.grid(row=5, column=1)
    typeOfBCa_combo.bind("<<ComboboxSelected>>", on_select_typeOfBcs)

    global scanned_selected
    global tyOfScanned_combo
    tyOfScanned = ["10X", "20X", "30X", "40X", "50X", "60X", "70X", "80X", "90X",]
    typeOfScanned_label = tk.Label(master=form_frame, text="Scanned:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    tyOfScanned_combo = ttk.Combobox(master=form_frame, values=tyOfScanned, font=("Verdana", 14), width=7)
    typeOfScanned_label.grid(row=6, column=0, sticky="e", padx=10, pady=10)
    tyOfScanned_combo.grid(row=6, column=1)
    tyOfScanned_combo.bind("<<ComboboxSelected>>", on_select_scanned)

    botton_frame = tk.Frame(master=left_frame, background='#222222', width=600, height=200)
    botton_frame.pack()

    empty_label = tk.Label(master=botton_frame, text='', height=3, bg='#222222')
    empty_label.grid(row=0, column=0)

    btn_submit = tk.Button(master=botton_frame, text="Submit", bg='#39b70c', command=submit_event)
    btn_submit.grid(row=1, column=0, padx=20, ipadx=20)
    btn_clear = tk.Button(master=botton_frame, text="Clear", bg='#d11919', command=delete_event)
    btn_clear.grid(row=1, column=1, ipadx=20)

    # RIGHT FRAME
    right_frame = tk.Frame(master=container_frame, bg='#323232', width=936, height=screen_height)
    right_frame.pack_propagate(False)
    right_frame.pack(side=tk.LEFT)

    empty_label = tk.Label(master=right_frame, text='', background='#323232', height=10)
    empty_label.pack()

    global folder_path
    open_button = tk.Button(master=right_frame, text='Open Folder', width=10, height=2, font=("Verdana", 13, "bold"), bg='#323232', fg='white', command=open_directory_dialog)
    open_button.pack(padx=20, pady=20)

    label = tk.Label(master=right_frame, text='Path folder:', font=("Verdana", 13, "bold"), bg='#323232', fg='white')
    label.pack(padx=20, pady=20)

    global path_label
    path_label = tk.Label(master=right_frame, text='', font=("Verdana", 9), bg='#323232', fg='white')
    path_label.pack(padx=20, pady=20)

    empty_label = tk.Label(master=right_frame, text='', background='#323232', height=10)
    empty_label.pack()

    prediction_button = tk.Button(master=right_frame, text='Prediction', width=15, height=3, font=("Verdana", 15, "bold"),  bg='#992a21', fg='#f4e9e8', command=prediction_to_whiling)
    prediction_button.pack(padx=30, pady=30)

    prediction_window.mainloop()



# Whiling Block
global save_img_at
global characters

save_img_at = 'E:/UDA_LEARNING/UDA_DO AN TRI TUE NHAN TAO & UNG DUNG/DO AN TRI TUE NHAN TAO & UNG DUNG/img'
characters = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))


def paths_to_dataframe():
    global data_frame
    global files
    global count_files
    file_paths = files
    num_files = count_files
    notification_label.config(text='Đang lấy dữ liệu...')
    data_frame = pd.DataFrame(index=np.arange(0, num_files), columns=['path', 'target'])
    k = 0
    for f in file_paths:
        data_frame.loc[k, 'path'] = f
        if re.findall(r'class0', f):
            data_frame.loc[k, 'target'] = 0
        elif re.findall(r'class1', f):
            data_frame.loc[k, 'target'] = 1
        k += 1
    
    global pos_paths
    global neg_paths
    pos_paths, neg_paths = data_to_Pos_Neg(data_frame)
    return data_frame


def data_to_Pos_Neg(data_frame):
    pos_paths = pd.DataFrame(data_frame.groupby("target").get_group(0)).reset_index(drop=True)
    neg_paths = pd.DataFrame(data_frame.groupby('target').get_group(1)).reset_index(drop=True)
    return pos_paths, neg_paths


def fig_class():
    files_to_dataFrame_thread.join()
    global data_frame
    global images_in_classes_path
    notification_label.config(text='Đang tính toán số lớp...')
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    class_counts = data_frame['target'].value_counts()
    class_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Số lượng ảnh của mỗi class')

    random_characters = np.random.choice(characters, size=10)
    random_characters = ''.join(random_characters)
    full_name = 'SoLuongAnhCuaMoiClass_' + random_characters + '.png'
    img_path = save_img_at + '/' + full_name
    fig.savefig(img_path)
    images_in_classes_path = img_path
    return img_path


def some_of_Neg():
    # hàm này không nên dùng
    files_to_dataFrame_thread.join()
    global neg_paths
    global images_in_class1_path
    notification_label.config(text='Đang lấy dữ liệu Negative...')
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
    images_in_class1_path = img_path
    return img_path


def some_of_Pos():
    # hàm này không nên dùng
    files_to_dataFrame_thread.join()
    global pos_paths
    global images_in_class0_path
    notification_label.config(text='Đang lấy dữ liệu Positive...')
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
    images_in_class0_path = img_path
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

def total_image_to_coordinate():
    files_to_dataFrame_thread.join()
    global pos_paths
    global neg_paths
    global coordinate_slice_path
    notification_label.config(text='Đang trực quan biểu đồ tọa độ các nhãn...')
    pos = path_to_coordinate_frame(pos_paths, 0)
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
    coordinate_slice_path = img_path
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


def tatal_image_to_slice():
    files_to_dataFrame_thread.join()
    global slice_path
    global pos_paths
    global neg_paths
    notification_label.config(text='Đang trực quan toàn bộ dữ liệu...')
    pos = path_to_coordinate_frame(pos_paths, 0)
    neg = path_to_coordinate_frame(neg_paths, 1)
    df_concat = pd.concat([pos, neg], axis=0)
    grid, mask, broken_patches = visualise_breast_tissue(df_concat)

    fig, ax = plt.subplots(1,2,figsize=(20, 10))
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
    slice_path = img_path
    return img_path


def total_slice_coordinate_predicted():
    finished_threads.join()

    global pos_paths
    global neg_paths
    global y_true
    global y_predict

    global total_image_to_coordinate_predicted_path
    pos = path_to_coordinate_frame(pos_paths, 0)
    neg = path_to_coordinate_frame(neg_paths, 1)
    df_concat = pd.concat([pos, neg], axis=0)
    
    for i in range(0, len(df_concat)):
        df_concat.loc[i, 'target'] = y_predict[i]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(df_concat.x.values, df_concat.y.values, c=df_concat.target.values, cmap='coolwarm', s=20);
    ax.set_title('Images on coordinate - predicted')
    ax.set_xlabel('x coord')
    ax.set_ylabel('y coord')

    random_characters = np.random.choice(characters, size=10)
    random_characters = ''.join(random_characters)
    full_name = 'Coordinate predicted_' + random_characters + '.png'
    img_path = save_img_at + '/' + full_name
    fig.savefig(img_path)
    total_image_to_coordinate_predicted_path = img_path


def join_threads():
    files_to_dataFrame_thread.join()
    get_images_in_classes_path_thread.join()
    images_in_class0_path_thread.join()
    images_in_class1_path_thread.join()
    coordinate_slice_path_thread.join()
    slice_path_thread.join()
    notification_label.config(text='Hoàn tất quá trình phân tích dữ liệu...1/3', fg='#18a92a')


def prediction_to_whiling():
    to_result()


def count_folders_and_files():
    global folders
    global files
    global count_folders
    global count_files
    global folder_path

    folders = []
    files = []
    count_folders = 0
    count_files = 0

    for dirname, _, filenames in os.walk(folder_path): # type: ignore
        folders.append(dirname)
        for filename in filenames:
            pathh = os.path.join(dirname, filename)
            files.append(pathh)

    count_folders = len(folders)
    count_files = len(files)
    return folders, files, count_folders, count_files


def read_and_preprocessing_data():
    '''đọc ảnh, chuẩn bị dữ liệu, chuẩn hóa'''
    join_thread.join()
    reading_label.config(text='Đang đọc dữ liệu...')
    global x_test
    global y_true
    global pos_paths
    global neg_paths

    WIDTH = 50
    HEIGHT = 50

    x_test = []
    y_true = []

    for idx in range(0, len(pos_paths)):
        org_img = imread(pos_paths.loc[idx, 'path'])
        resized_img = resize(org_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        x_test.append(resized_img)
        y_true.append(0)

    for idx in range(0, len(neg_paths)):
        org_img = imread(neg_paths.loc[idx, 'path'])
        resized_img = resize(org_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        x_test.append(resized_img)
        y_true.append(1)

    # combined = list(zip(x_test, y_true))
    # np.random.shuffle(combined)
    # x_test, y_true = zip(*combined)

    reading_label.config(text='Đang chuẩn hóa dữ liệu...')
    x_test = np.array(x_test)/255.0
    y_true = np.array(y_true)

    reading_label.config(text='Hoàn tất quá trình đọc và xử lí dữ liệu...2/3', fg='#18a92a')


def prediction():
    '''thực hiện predict trên mỗi ảnh'''
    read_and_preprocessing_data_thread.join()
    finished_label.config(text='Đang tải model...')
    
    transformer_model_path = 'E:/UDA_LEARNING/UDA_DO AN TRI TUE NHAN TAO & UNG DUNG/DO AN TRI TUE NHAN TAO & UNG DUNG/src/BreastCancerModel_Transformer_checkpoint.weights.h5'
    CNN_model_path = 'E:/UDA_LEARNING/UDA_DO AN TRI TUE NHAN TAO & UNG DUNG/DO AN TRI TUE NHAN TAO & UNG DUNG/src/BreastCancerModel_CNN.keras'
    MobileNet_model_path = 'E:/UDA_LEARNING/UDA_DO AN TRI TUE NHAN TAO & UNG DUNG/DO AN TRI TUE NHAN TAO & UNG DUNG/src/BreastCancerModel_MobileNet.keras'

    model = load_model(CNN_model_path)
    time.sleep(1)

    global x_test
    global y_true
    global y_predict

    y_predict = []
    finished_label.config(text='Đang thực hiện predict...')

    predicted = model.predict(x_test) # type: ignore
    # print(predicted)

    for i in predicted:
        if i >= 0.5:
            y_predict.append(1)
        else:
            y_predict.append(0)

    # print(y_predict)
    # print(y_true)

    conf_matrix = confusion_matrix(y_true, y_predict)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    random_characters = np.random.choice(characters, size=10)
    random_characters = ''.join(random_characters)
    full_name = 'Confusion matrix_' + random_characters + '.png'
    img_path = save_img_at + '/' + full_name
    plt.savefig(img_path)
    global conf_matrix_path
    conf_matrix_path = img_path

    global accuracy
    global precision
    global recall
    global f1

    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)

    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')

    finished_label.config(text='Hoàn tất quá trình predict...3/3', fg='#18a92a')
    view_result.pack(padx=20, pady=40)


def finished_all_threads():
    '''đợi hai luồng trên xong thì chạy thông báo kết quả đã xong!'''
    prediction_thread.join()
    

def window_whiling():
    global whiling_window
    whiling_window = tk.Tk()
    whiling_window.title("Đang phân tích kết quả")

    # CONTAINER FRAME
    container_frame = tk.Frame(master=whiling_window, bg='#323232', width=600, height=600)
    container_frame.pack_propagate(False)
    container_frame.pack(fill=tk.BOTH, expand=True)

    label = tk.Label(master=container_frame, text='Đang phân tích kết quả...', 
                    font=("Verdana", 15, 'bold'),
                    bg='#323232',
                    fg='white',
                    padx=60,
                    pady=60)
    label.pack()

    count_frame = tk.Frame(master=container_frame, bg='#323232')
    count_frame.pack()
    folders, files, count_folders, count_files = count_folders_and_files()

    folders_label = tk.Label(master=count_frame, text='Folders:', font=('Verdana', 13), bg='#323232', fg='white',)
    count_folders_label = tk.Label(master=count_frame, text=str(count_folders), font=('Verdana', 13), bg='#323232', fg='white',)
    folders_label.grid(row=0, column=0, sticky="e", padx=10, pady=10)
    count_folders_label.grid(row=0, column=1, padx=15, pady=10)

    files_label = tk.Label(master=count_frame, text='Files:', font=('Verdana', 13), bg='#323232', fg='white',)
    count_files_label = tk.Label(master=count_frame, text=str(count_files), font=('Verdana', 13), bg='#323232', fg='white',)
    files_label.grid(row=1, column=0, sticky="e", padx=10, pady=10)
    count_files_label.grid(row=1, column=1, padx=15, pady=10)

    space_label = tk.Label(master=container_frame, text='', font=('Verdana', 13), bg='#323232', fg='white')
    space_label.pack(padx=10, pady=20)

    global notification_label
    notification_label = tk.Label(master=container_frame, text='Đang xử lí...', font=('Verdana', 13), bg='#323232', fg='white')
    notification_label.pack(padx=30, pady=10)

    global reading_label
    global finished_label
    reading_label = tk.Label(master=container_frame, text='', font=('Verdana', 13), bg='#323232', fg='white')
    reading_label.pack(padx=30, pady=10)
    finished_label = tk.Label(master=container_frame, text='', font=('Verdana', 13), bg='#323232', fg='white')
    finished_label.pack(padx=30, pady=10)
    
    # QÚA TRÌNH PHÂN TÍCH DỮ LIỆU - ĐỌC - TIỀN XỬ LÍ DỮ LIỆU ĐẦU VÀO - CHUẨN HÓA DỮ LIỆU - PREDICT TRÊN MODEL

    # data frame (một frame chứa các paths của file và target cho file đó)
    # đường dẫn ảnh số lượng ảnh của mỗi lớp
    # đường dẫn ảnh một vài ví dụ về mẫu âm tính (-)
    # đường dẫn ảnh một vài ví dụ về mẫu dương tính (+)
    # đường dẫn ảnh tọa độ mỗi ảnh của toàn bộ slice
    # đường dẫn ảnh ghép toàn bộ slice và toàn bộ slice được đánh dấu dương tính

    # đường dẫn ảnh ma trận hỗn loạn sau khi predict
    # đường dẫn ảnh tọa độ mỗi ảnh của toàn bộ slice sau khi predict
    # đường dẫn ảnh ghép toàn bộ slice được đánh dấu dương tính sau khi predict

    global id_patient
    global data_frame
    global images_in_classes_path
    global images_in_class0_path
    global images_in_class1_path
    global coordinate_slice_path
    global slice_path
    global conf_matrix_path

    global confusion_matrix_path_predicted
    global slice_path_predicted
    global coordinate_slice_path_predicted

    global files_to_dataFrame_thread
    global get_images_in_classes_path_thread
    global images_in_class0_path_thread
    global images_in_class1_path_thread
    global coordinate_slice_path_thread
    global slice_path_thread
    global join_thread

    global pos_paths
    global neg_paths

    files_to_dataFrame_thread = threading.Thread(target=paths_to_dataframe, name='files_to_dataFrame_thread')
    files_to_dataFrame_thread.start()

    get_images_in_classes_path_thread = threading.Thread(target=fig_class, name='get_images_in_classes_path_thread')
    get_images_in_classes_path_thread.start()

    images_in_class0_path_thread = threading.Thread(target=some_of_Pos, name='images_in_class0_path_thread')
    images_in_class0_path_thread.start()

    images_in_class1_path_thread = threading.Thread(target=some_of_Neg, name='images_in_class1_path_thread')
    images_in_class1_path_thread.start()

    coordinate_slice_path_thread = threading.Thread(target=total_image_to_coordinate, name='coordinate_slice_path_thread')
    coordinate_slice_path_thread.start()

    slice_path_thread = threading.Thread(target=tatal_image_to_slice, name='slice_path_thread')
    slice_path_thread.start()

    join_thread = threading.Thread(target=join_threads, name='joind_thread')
    join_thread.start()


    global CNN_v1_model
    global CNN_v2_model
    global MobileNet_v1_model
    global MobileNet_v2_model
    global EffientNet_model
    global Transformer_model
    global ShuffleNet_model

    global read_and_preprocessing_data_thread
    global prediction_thread
    
    read_and_preprocessing_data_thread = threading.Thread(target=read_and_preprocessing_data, name='read_and_preprocessing_data_thread')
    read_and_preprocessing_data_thread.start()

    prediction_thread = threading.Thread(target=prediction, name='prediction_thread')
    prediction_thread.start()

    global finished_threads
    finished_threads = threading.Thread(target=finished_all_threads, name='finished_all_threads')
    finished_threads.start()

    # vẽ lại ảnh tọa độ sau khi predict
    total_slice_coordinate_predicted_thread = threading.Thread(target=total_slice_coordinate_predicted, name='total_slice_coordinate_predicted_thread')
    total_slice_coordinate_predicted_thread.start()

    global view_result
    view_result = tk.Button(master=container_frame, text='Xem kết quả', width=15, height=1, font=("Verdana", 10),  bg='#39b70c', command=window_result)
    # view_result.pack(padx=20, pady=20)

    whiling_window.mainloop()



# Result Block
def back_process():
    result_window.destroy()
    global folder_path
    folder_path = ''
    window_prediction()


def on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")


def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox('all'))


def window_result():
    whiling_window.destroy()
    global result_window
    result_window = tk.Tk()
    result_window.title('Kết quả phân tích (Biểu mô ống xâm lấn - IDC)')
    screen_width = result_window.winfo_screenwidth()
    screen_height = result_window.winfo_screenheight()

    # CONTAINER FRAME
    container_frame = tk.Frame(master=result_window, bg='#323232', width=screen_width, height=screen_height)
    container_frame.pack_propagate(False)
    container_frame.pack(fill=tk.BOTH, expand=True)

    # LEFT FRAME
    left_frame = tk.Frame(master=container_frame, background='#222222', width=600, height=screen_height, border=3)
    left_frame.pack_propagate(False)
    left_frame.pack(anchor="nw", side=tk.LEFT)
    label = tk.Label(master=left_frame,
                    text="Kết quả phân tích hồ sơ (BCa - IDC)",
                    font=("Verdana", 15, 'bold'),
                    bg='#222222',
                    fg='white',
                    padx=60,
                    pady=60)
    label.pack()
    empty_frame = tk.Frame(master=left_frame, width= 600, height=1, bg='yellow')
    empty_frame.pack()

    # FORM FRAME
    form_frame = tk.Frame(master=left_frame, width= 600, height=400, bg='#222222')
    form_frame.pack()

    global first_name
    global last_name
    global id_patient
    global age
    global gender
    global typeOfBCa
    global scanned

    first_name_label = tk.Label(master=form_frame, text="First Name:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    first_name_ = tk.Label(master=form_frame, text="" + first_name, height=1, font=("Verdana", 14), fg='white', bg='#222222')
    first_name_label.grid(row=0, column=0, sticky="e", padx=10, pady=10)
    first_name_.grid(row=0, column=1, padx=15, pady=10)

    last_name_label = tk.Label(master=form_frame, text="Last Name:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    last_name_ = tk.Label(master=form_frame, text="" + last_name, height=1, font=("Verdana", 14), fg='white', bg='#222222')
    last_name_label.grid(row=1, column=0, sticky="e", padx=10, pady=10)
    last_name_.grid(row=1, column=1, padx=15, pady=10)

    id_patient_label = tk.Label(master=form_frame, text="Patient ID:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    id_patient_ = tk.Label(master=form_frame, text="" + id_patient, height=1, font=("Verdana", 14), fg='white', bg='#222222')
    id_patient_label.grid(row=2, column=0, sticky="e", padx=10, pady=10)
    id_patient_.grid(row=2, column=1, padx=15, pady=10)

    age_label = tk.Label(master=form_frame, text="Age:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    age_ = tk.Label(master=form_frame, text="" + age, height=1, font=("Verdana", 14), fg='white', bg='#222222')
    age_label.grid(row=3, column=0, sticky="e", padx=10, pady=10)
    age_.grid(row=3, column=1, padx=15, pady=10)

    gender_label = tk.Label(master=form_frame, text="Gender", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    gender_ = tk.Label(master=form_frame, text="" + gender, height=1, font=("Verdana", 14), fg='white', bg='#222222')
    gender_label.grid(row=4, column=0, sticky="e", padx=10, pady=10)
    gender_.grid(row=4, column=1)

    typeOfBCa_label = tk.Label(master=form_frame, text="Type of BCa:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    typeOfBCa_ = tk.Label(master=form_frame, text="" + typeOfBCa, height=1, font=("Verdana", 14), fg='white', bg='#222222')
    typeOfBCa_label.grid(row=5, column=0, sticky="e", padx=10, pady=10)
    typeOfBCa_.grid(row=5, column=1)

    typeOfScanned_label = tk.Label(master=form_frame, text="Scanned:", height=1, font=("Verdana", 14), fg='white', bg='#222222')
    tyOfScanned_ = tk.Label(master=form_frame, text="" + scanned, height=1, font=("Verdana", 14), fg='white', bg='#222222')
    typeOfScanned_label.grid(row=6, column=0, sticky="e", padx=10, pady=10)
    tyOfScanned_.grid(row=6, column=1)

    botton_frame = tk.Frame(master=left_frame, background='#222222', width=600, height=200)
    botton_frame.pack()

    empty_label = tk.Label(master=botton_frame, text='', height=3, bg='#222222')
    empty_label.grid(row=0, column=0)

    btn_submit = tk.Button(master=botton_frame, text="Xuất PDF", bg='#39b70c', command=submit_event)
    btn_submit.grid(row=1, column=0, padx=20, ipadx=20)
    btn_clear = tk.Button(master=botton_frame, text="Quay lại", bg='yellow', command=back_process)
    btn_clear.grid(row=1, column=1, ipadx=20)

    # RIGHT FRAME
    right_frame = tk.Frame(master=container_frame, bg='#323232', width=936, height=screen_height)
    right_frame.pack_propagate(False)
    right_frame.pack(side=tk.LEFT)

    # image paths
    global images_in_class_path
    global images_in_class0_path
    global images_in_class1_path
    global coordinate_slice_path
    global slice_path
    global conf_matrix_path
    global total_image_to_coordinate_predicted_path

    # open files
    images_in_class = Image.open(images_in_classes_path)
    images_in_class0 = Image.open(images_in_class0_path)
    images_in_class1 = Image.open(images_in_class1_path)
    coordinate_slice = Image.open(coordinate_slice_path)
    slice_image = Image.open(slice_path)
    conf_matrix_image = Image.open(conf_matrix_path)
    total_image_to_coodinate_predicted = Image.open(total_image_to_coordinate_predicted_path)

    # resize ảnh trước khi chuyển sang đối tượng tkinter
    images_in_class = images_in_class.resize((450, 450), Image.Resampling.LANCZOS)
    images_in_class0 = images_in_class0.resize((900, 500), Image.Resampling.LANCZOS)
    images_in_class1 = images_in_class1.resize((900, 500), Image.Resampling.LANCZOS)
    coordinate_slice = coordinate_slice.resize((450, 450), Image.Resampling.LANCZOS)
    slice_image = slice_image.resize((900, 500), Image.Resampling.LANCZOS)
    conf_matrix_image = conf_matrix_image.resize((450, 450), Image.Resampling.LANCZOS)
    total_image_to_coodinate_predicted = total_image_to_coodinate_predicted.resize((450, 450), Image.Resampling.LANCZOS)

    # convert to Tkinter_image format
    images_in_class_tk = ImageTk.PhotoImage(images_in_class)
    images_in_class0_tk = ImageTk.PhotoImage(images_in_class0)
    images_in_class1_tk = ImageTk.PhotoImage(images_in_class1)
    coordinate_slice_tk = ImageTk.PhotoImage(coordinate_slice)
    slice_image_tk = ImageTk.PhotoImage(slice_image)
    conf_matrix_image_tk = ImageTk.PhotoImage(conf_matrix_image)
    total_image_to_coodinate_predicted_tk = ImageTk.PhotoImage(total_image_to_coodinate_predicted)


    # tạo scrollbar
    scrollbar = tk.Scrollbar(master=right_frame, orient='vertical', bg='#323232')
    scrollbar.pack(side='right', fill='y')

    # canvas chỉ được dùng trong phạm vi cửa sổ kết quả!
    global canvas 
    canvas = tk.Canvas(master=right_frame, yscrollcommand=scrollbar.set, bg='#323232')
    canvas.pack(side="top", fill="both", expand=True)
    scrollbar.config(command=canvas.yview)
    inner_frame = tk.Frame(canvas, bg='#323232')
    canvas.create_window((0, 0), window=inner_frame, anchor='nw')
    inner_frame.bind("<Configure>", on_configure)
    canvas.bind('<Configure>', on_configure)
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    # THÊM CÁC LABEL VÀ HÌNH ẢNH Ở ĐÂY!
    frame0 = tk.Frame(master=inner_frame)
    frame0.pack()
    label01 = tk.Label(master=frame0, text='Số lượng ảnh của mỗi class', font=('Verdana', 14))
    label01.grid(row=0, column=0, padx=1, pady=20)
    label02 = tk.Label(master=frame0, text='Ma trận hỗn loạn', font=('Verdana', 14))
    label02.grid(row=0, column=1, padx=1, pady=20)
    images_in_class_tk_label = tk.Label(master=frame0, image=images_in_class_tk)
    images_in_class_tk_label.grid(row=1, column=0)
    conf_matrix_image_tk_label = tk.Label(master=frame0, image=conf_matrix_image_tk)
    conf_matrix_image_tk_label.grid(row=1, column=1)

    frame1 = tk.Frame(master=inner_frame)
    frame1.pack()
    label11 = tk.Label(master=frame1, text='Mẫu ảnh trên toàn bộ slice', font=('Verdana', 14))
    label11.pack(padx=1, pady=20)
    slice_image_tk_label = tk.Label(master=frame1, image=slice_image_tk)
    slice_image_tk_label.pack()

    frame2 = tk.Frame(master=inner_frame)
    frame2.pack()
    label21 = tk.Label(master=frame2, text='Ảnh tạo độ nhị phân ban đầu', font=('Verdana', 14))
    label21.grid(row=0, column=0, padx=1, pady=20)
    label22 = tk.Label(master=frame2, text='Ảnh tọa độ nhị phân sau predict', font=('Verdana', 14))
    label22.grid(row=0, column=1, padx=1, pady=20)
    coordinate_slice_tk_label = tk.Label(master=frame2, image=coordinate_slice_tk)
    coordinate_slice_tk_label.grid(row=1, column=0)
    total_image_to_coodinate_predicted_tk_label = tk.Label(master=frame2, image=total_image_to_coodinate_predicted_tk)
    total_image_to_coodinate_predicted_tk_label.grid(row=1, column=1)

    frame3 = tk.Frame(master=inner_frame)
    frame3.pack()
    label31 = tk.Label(master=frame3, text='Hình ảnh các mẫu bệnh dương tính', font=('Verdana', 14))
    label31.pack(padx=1, pady=20)
    images_in_class1_tk_label = tk.Label(master=frame3, image=images_in_class1_tk)
    images_in_class1_tk_label.pack()
    label32 = tk.Label(master=frame3, text='Hình ảnh các mẫu bệnh âm tính', font=('Verdana', 14))
    label32.pack(padx=1, pady=20)
    images_in_class0_tk_label = tk.Label(master=frame3, image=images_in_class0_tk)
    images_in_class0_tk_label.pack()

    result_window.mainloop()


# Welcome Block
def check_button_event_to_prediction():
    welcome_window.destroy()
    window_prediction()

def window_welcome():
    global welcome_window
    welcome_window = tk.Tk()
    welcome_window.title('Chào mừng!')
    frame_welcome = tk.Frame(bg='#323232', width=900, height=570)
    frame_welcome.pack(fill=tk.BOTH, expand=True)

    label_welcome = tk.Label(master=frame_welcome, text='Welcome!', font=("Verdana", 25, "bold"), bg='#323232', fg='white', anchor='center', width=35, height=4)
    label_welcome.pack()
    
    label_check = tk.Label(master=frame_welcome, text='Phát hiện tế bào ung thư bên dưới!', font=("Verdana", 13), bg='#323232', fg='white', anchor='center', height=4)
    label_check.pack()

    check_button = tk.Button(master=frame_welcome, text='Kiểm tra ngay', width=13, height=2, bg='#992a21', fg='#f4e9e8', font=("Verdana", 17, 'bold'), borderwidth=4, command=check_button_event_to_prediction)
    check_button.pack()

    sapce_label = tk.Label(master=frame_welcome, text='', height=4, bg='#323232')
    sapce_label.pack()

    login_label = tk.Label(master=frame_welcome, text='Log in', font=("Verdana", 10, "underline",), height=2, bg='#323232', fg='white')
    login_label.pack()
    Signup_label = tk.Label(master=frame_welcome, text='Sign up', font=("Verdana", 10, "underline"), height=2, bg='#323232', fg='white')
    Signup_label.pack()

    sapce_label = tk.Label(master=frame_welcome, text='', height=4, bg='#323232')
    sapce_label.pack()

    other_labels = [
        'Xem hướng dẫn',
        'Chính sách của chúng tôi',
        'Giấy phép và điều khoản',
        'Liên hệ với chúng tôi'
    ]
    for i in range(0, len(other_labels)):
        label_1 = tk.Label(master=frame_welcome, text=other_labels[i], font=("Verdana", 10), bg='#323232', fg='white', anchor='center', height=5)
        label_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    welcome_window.mainloop()


global first_name
global last_name
global id_patient
global age
global gender
global typeOfBCa
global scanned

global first_name_entry
global last_name_entry
global id_patient_entry
global age_entry
global gender_selected
global typeOfBCa_selected
global scanned_selected

global gender_combo
global typeOfBCa_combo
global tyOfScanned_combo
global folder_path
folder_path = ''

window_welcome()