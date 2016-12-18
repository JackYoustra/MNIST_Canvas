'''
Created on Dec 15, 2016

@author: jack
'''

import tkinter as tk
from PIL import Image, ImageDraw
import DeepTutorial

width = 400
height = 400

class Application(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, width=width, height=height, cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        #self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_button_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # create draw stuff
        self.memImage = Image.new("L", (width, height), "white") #fill with white
        self.draw = ImageDraw.Draw(self.memImage)
        
        self.after(500, func=self.train)

    def on_button_press(self, event):
        self.x = event.x
        self.y = event.y

    def on_button_move(self, event):
        kernel_size = 10
        self.canvas.create_rectangle(event.x-kernel_size, event.y-kernel_size, event.x+kernel_size, event.y+kernel_size, fill="black")
        self.draw.rectangle([event.x-kernel_size, event.y-kernel_size, event.x+kernel_size, event.y+kernel_size], fill="black")
        
    def on_button_release(self, event):
        #x0,y0 = (self.x, self.y)
        #x1,y1 = (event.x, event.y)
        DeepTutorial.predict(self.memImage)
        # reset canvas n stuff
        self.memImage = Image.new("L", (width, height), "white") #fill with white
        self.draw = ImageDraw.Draw(self.memImage)
        self.canvas.delete("all")
        #self.canvas.create_rectangle(x0,y0,x1,y1, fill="black")
        #self.memImage.save("tmp.jpg")
        
    def train(self):
        DeepTutorial.trainModel()
#        self.task()
    
    def task(self):
        self.after(2000, func=self.task)  # reschedule event in 2 seconds

app = Application()
app.mainloop()
