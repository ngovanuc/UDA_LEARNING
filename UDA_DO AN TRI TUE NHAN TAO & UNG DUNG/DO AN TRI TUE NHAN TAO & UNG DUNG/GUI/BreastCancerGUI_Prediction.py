"""
Author: Ngo Van Uc
Gmail: ngovanuc.1508@gmail.com
License: (contact to Ngo Van Uc)
"""

import os
import tkinter as tk
import tkinter.ttk as ttk

from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askdirectory

from BreastCancerGUI_Welcome import *
from BreastCancerGUI_Whiling import *
from BreastCancerGUI_Result import *
from BreastCancerGUI import *


def on_select_gender(event):
    selected_value = gender_combo.get()
    gender = selected_value
    print("Selected value:", gender)

def on_select_typeOfBcs(event):
    selected_value = typeOfBCa_combo.get()
    typeOfBCa = selected_value
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
    first_name = first_name_entry.delete(0, tk.END)
    last_name = last_name_entry.delete(0, tk.END)
    id_patient = id_patient_entry.delete(0, tk.END)
    age = age_entry.delete(0, tk.END)
    gender = gender_combo.delete(0, tk.END)
    typeOfBCa = typeOfBCa_combo.delete(0, tk.END)
    scanned = tyOfScanned_combo.delete(0, tk.END)

    confirm_clear()

    print(first_name)
    print(last_name)
    print(id_patient)
    print(age)
    print(gender)
    print(typeOfBCa)
    print(scanned)

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

def predict_to_result():
    try:
        if os.path.exists(folder_path): # type: ignore
            print("Đường dẫn tồn tại: ", folder_path)
            if os.path.isdir(folder_path): # type: ignore
                print("Thư mục hợp lệ: ", folder_path)
                prediction_window.destroy()
                window_result()
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

    prediction_button = tk.Button(master=right_frame, text='Prediction', width=15, height=3, font=("Verdana", 15, "bold"),  bg='#992a21', fg='#f4e9e8', command=predict_to_result)
    prediction_button.pack(padx=30, pady=30)

    prediction_window.mainloop()