#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:48:56 2020

@author: sidu
"""
#!/usr/bin/python
import sys
import os
import comparatorclass
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile 



root=Tk()
root.geometry('1000x1000')  


def open_file(): 
    file = askopenfile(mode ='r', filetypes =[('Python Files', '*.py')]) 
    if file is not None: 
        content = file.read() 
        print(content) 

def run_comparator():
    comparatorclass.main()
    

button2 = Button(root, text ='Open', command = lambda:open_file()) 
button2.pack(side = TOP, pady = 10) 

button1 = Button(root,text="hello",command= run_comparator)
button1.pack()




root.mainloop()