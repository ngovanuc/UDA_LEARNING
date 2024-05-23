"""
Author: Ngo Van Uc
Gmail: ngovanuc.1508@gmail.com
License: (contact to Ngo Van Uc)
"""

import os
import glob
import time
import tkinter as tk
import tkinter.ttk as ttk

# from BreastCancerGUI_Welcome import *
# from BreastCancerGUI_Prediction import *
# from BreastCancerGUI_Result import *
# from BreastCancerGUI import *


global path
path = 'E:/UDA_LEARNING/UDA_DO AN TRI TUE NHAN TAO & UNG DUNG/BREAST CANCER/data/IDC_regular_ps50_idx5/8864'



# Whiling Block
def count_folders_and_files():
    global folders
    global files
    global count_folders
    global count_files

    folders = []
    files = []

    count_folders = 0
    count_files = 0

    for dirname, _, filenames in os.walk(path):
        folders.append(dirname)
        for filename in filenames:
            pathh = os.path.join(dirname, filename)
            files.append(pathh)

    count_folders = len(folders)
    count_files = len(files)

    return folders, files, count_folders, count_files

    
def window_whiling():
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
    


    # quá trình tiến hành xử lí kết quả nó hiện ở đây!
    # path_label = tk.Label(master=container_frame, text='', font=("Verdana", 9), bg='#323232', fg='white', wraplength=550)
    # path_label.pack(padx=20, pady=50)
    # for idx, f in enumerate(files[0:5]):
    #     path_label.config(text='' + f)
        

    whiling_window.mainloop()
        

window_whiling()
