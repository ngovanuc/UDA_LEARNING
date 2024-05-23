"""
Author: Ngo Van Uc
Gmail: ngovanuc.1508@gmail.com
License: (contact to Ngo Van Uc)
"""

import os
import glob
import tkinter as tk
import tkinter.ttk as ttk

from BreastCancerGUI_Welcome import *
from BreastCancerGUI_Prediction import *
from BreastCancerGUI_Whiling import *
from BreastCancerGUI import *


def window_result():
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
    btn_clear = tk.Button(master=botton_frame, text="Quay lại", bg='yellow', command=delete_event)
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

    prediction_button = tk.Button(master=right_frame, text='Prediction', width=15, height=3, font=("Verdana", 15, "bold"),  bg='#992a21', fg='#f4e9e8', command=predict_to_result) # type: ignore
    prediction_button.pack(padx=30, pady=30)

    result_window.mainloop()
