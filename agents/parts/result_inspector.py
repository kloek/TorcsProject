import tkinter as tk
from tkinter import filedialog
import numpy as np


# open a file dialog to pick file!!!
root = tk.Tk()
root.withdraw()
filename = filedialog.askopenfilename()

print("selected file is = " + str(filename))

# if file is npy ...
data = np.load(file=filename)

print("data.shape= " + str(data.shape))
print("data= " + str(data))



