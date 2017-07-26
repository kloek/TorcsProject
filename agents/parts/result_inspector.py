import tkinter as tk
from tkinter import filedialog
import numpy as np
import math
import matplotlib.pyplot as plt

# episode, total_steps, best_total_reward, total_reward, r_t, epsilon
# script assumes you now what you are doing, not much error checking!!!!
COL_NAMES = ["episode", "total_steps", "best_total_reward", "total_reward", "r_t", "epsilon"]

#y_col = None
#x_col = None


# subrutine to main, of for automaticly creating plots
def create_plot(npy_filename, y_col, x_col, title, remove_first=False):
    print("selected file is = " + str(npy_filename))

    #TODO if file is npy ...
    data = np.load(file=npy_filename)

    if(remove_first): #removes first line in data since it is start values, and not results!
        data = np.delete(data, (0), axis=0)

    print("data.shape= " + str(data.shape))
    #print("data= " + str(data))

    print("y:" + str(y_col) + " x:" + str(x_col))

    print("data_y= " + str(data[:,y_col]))
    print("data_x= " + str(data[:,x_col]))

    plt.plot(data[:,x_col],data[:,y_col])

    plt.ylabel(COL_NAMES[y_col])
    plt.xlabel(COL_NAMES[x_col])
    plt.title(title)

    plt.savefig(filename + "_" + title + ".jpg")



# for manually creating plots....
if __name__ == "__main__":
    remove_first=False
    # open a file dialog to pick file!!!
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()

    # select y and x columns
    y_col = int(input("enter a number for y column \n [episode, total_steps, best_total_reward, total_reward, r_t, epsilon]: "))
    x_col = int(input("enter a number for x column \n [episode, total_steps, best_total_reward, total_reward, r_t, epsilon]: "))

    title = input("Select a title for plot (blank will generate title):")

    if(title == ""):
        # generate a title
        title = "PLOT_y" + str(y_col) + "_x" + str(x_col)

    if(y_col == 2):
        remove_first=True

    create_plot(npy_filename=filename, y_col=y_col, x_col=x_col, title=title, remove_first=remove_first)
