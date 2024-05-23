"""
Author: Ngo Van Uc
Gmail: ngovanuc.1508@gmail.com
License: (contact to Ngo Van Uc)
"""

import tkinter as tk
import tkinter.ttk as ttk

from BreastCancerGUI_Prediction import *
from BreastCancerGUI_Whiling import *
from BreastCancerGUI_Result import *
from BreastCancerGUI import *


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