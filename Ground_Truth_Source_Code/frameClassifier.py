import cv2
import csv
import numpy as np
import os
import threading
from PIL import Image, ImageTk


import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
from matplotlib import pyplot as plt

import tkinter as tk
from tkinter import Tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import Canvas

LARGE_FONT = ("Verdana", 12)
style.use("ggplot")

allframes = []

framestep = 2
startingframe = 300

class mainUI(tk.Tk):
   
    def __init__(self, *args, **kwargs): #args = any number of variables, kwargs = pass throguh dictionary
        
        tk.Tk.__init__(self, *args, **kwargs)
        #anaylsis = Analysis()
        
        #tk.Tk.iconbitmap(self, default="clienticon.ico") #change logo in the top cornor
        tk.Tk.wm_title(self, "Kevin retinopathy project")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        #menubar
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        graphmenu = tk.Menu(menubar,tearoff=0)
        
        self.graphOption = tk.StringVar()
        self.graphOption.set("1")

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, PageOne):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column = 0, sticky="nsew")

        filemenu.add_command(label="Load", command = lambda: self.frames[StartPage].getVideo())
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        self.show_frame(StartPage)


    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
    


class StartPage(tk.Frame):


    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font = LARGE_FONT)
        label.pack(pady=10, padx=10)

        self.canvas = None
        self.toolbar = None
        self.r = None
        self.converting = False
        self.displayText = "testing"
        self.displayframe = "0"
        self.display = tk.Label(self, text = self.displayText)
        #self.counting = tk.DoubleVar(self, value=0)
        self.display2 = tk.Label(self, text = self.displayframe)
        #self.text1 = tk.Text(self, height = 10, width = 50)
        #self.text1.insert(tk.INSERT, self.displayText)
        #text1.config(state="disabled")
        self.display2.pack()
        self.display.pack()
        self.x = self.y = 0
        self.finalImg = None

        self.f = None
        self.a = None

        self.currentFrame = startingframe
        self.count = 0

        self.frames = []


        self.sharpFrames = []

        self.blurFrames = []


        button5= ttk.Button(self, text="print sharp", 
                             command=lambda: self.printSharp())
        button5.pack(side="left", anchor="s")

        button1 = ttk.Button(self, text="clear", 
                             command=lambda: self.clearImg())
        button1.pack(side="left", anchor="s" )

        button6 = ttk.Button(self, text="save CSV", 
                             command=lambda: self.saveAsCSV())
        button6.pack(side="left", anchor="s" )

        

        button5= ttk.Button(self, text="ReadCSV", 
                             command=lambda: self.readAsCSV())
        button5.pack(side="left", anchor="s")






        button2 = ttk.Button(self, text="Display img", 
                             command=lambda: self.displayImg())
        button2.pack(side="right", anchor="s", padx=20)

        button3= ttk.Button(self, text="Sharp", 
                             command=lambda: self.sharpImg())
        button3.pack(side="right", anchor="s")

        button4= ttk.Button(self, text="Blur", 
                             command=lambda: self.blurImg())
        button4.pack(side="right", anchor="s")
        
        controller.bind('q', lambda event: self.blurImg())
        controller.bind('w', lambda event: self.sharpImg())
        controller.bind('e', lambda event: self.displayImg())

        # button3 = ttk.Button(self, text="drew rec", 
        #                      command=lambda: self.drawRec())
        # button3.pack()
    
    def fileExplorer(self):
        #text.config(state="enable")
        print("call fileExplorer")
        self.file_path = filedialog.askopenfilename()
        if not self.file_path:
            print("cancelled")
            self.converting = False
            self.file_path = None
            return
        self.display.configure(text=self.file_path)
        allframes.clear()
        self.frames.clear()
        self.sharpFrames.clear()
        self.blurFrames.clear()
        self.sharpCount = 0
        self.blurCount = 0
        self.currentFrame = startingframe
        #self.clearGraph()
        if self.canvas != None:
            self.canvas.delete("all")
        

        currentFrame = 0
        print(self.file_path)
        cap = cv2.VideoCapture(self.file_path)
        self.converting = True

        while(True):
            ret, frame = cap.read()
            if not ret:
                break

            
            allframes.append(frame)

            name = str(currentFrame)
            print("Extracting frame..." + name)
            self.display2.configure(text=name) 

            currentFrame += 1 

        self.converting = False
        
        if self.canvas == None:
            self.canvas = Canvas(self,  width=800, height=800)
            self.canvas.pack() 
        print(len(allframes))

    def getVideo(self):
        if(self.converting == False):
            self.converting = True
            t2 = threading.Thread(target=self.fileExplorer)
            t2.start()
    
    def clearImg(self):
        self.canvas.delete("all")

    def displayImg(self, event=None):
        
        self.canvas.delete("all")
        test = self.currentFrame
        if(test + framestep < len(allframes)):
            self.currentFrame += framestep
            img = allframes[self.currentFrame]
            #crop_img = img[300:600, 300:600]
            colorImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(colorImg.shape)
            h, w, cc = img.shape
            
            tk_im = Image.fromarray(colorImg)
            self.finalImg = ImageTk.PhotoImage(tk_im)
            self.canvas.create_image(0, 0, anchor="nw", image=self.finalImg)
            
            print("sharp number: " + str(self.sharpCount))
            print("blur number: " + str(self.blurCount))
            print("Current frame: "+ str(self.currentFrame))
            

    def sharpImg(self, event=None):
        if(self.currentFrame < len(allframes)):
            self.sharpFrames.append(self.currentFrame)
            self.sharpCount += 1

    def blurImg(self, event=None):
        if(self.currentFrame < len(allframes)):
            self.blurFrames.append(self.currentFrame)
            self.blurCount += 1
    
    def printSharp(self):
        print(self.sharpFrames)

    def saveAsCSV(self):
        self.save_file_path = filedialog.asksaveasfilename()
        print(self.save_file_path)
        # data = [[100, 24, 46],
        #         [120, 33]]

        # for i in data:
        #     print(i[0])
        data = []
        data.append(self.sharpFrames)
        data.append(self.blurFrames)

        with open(self.save_file_path, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)

    def readAsCSV(self):
        csv_file_path = filedialog.askopenfilename()
        print(csv_file_path)

        with open(csv_file_path) as f:
            reader = csv.reader(f)
            sharp = next(reader)
            blur = next(reader)

        print(len(sharp))
        print(len(blur))



    # def drawRec(self):
    #     self.canvas.create_rectangle(50, 25, 150, 75, fill="blue")

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="page one", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        self.x = self.y = 0
        self.canvas = None
        self.finalImg = None

        self.rect = None
        self.start_x = None
        self.start_y = None 

        # button1 = ttk.Button(self, text="Show Pic", 
        #                     command=lambda: self.displayImg())
        # button1.pack()

        button2 = ttk.Button(self, text="Save", 
                            command=lambda: controller.show_frame(StartPage))
        button2.pack()


app = mainUI()
app.geometry("1280x1024")
app.mainloop()