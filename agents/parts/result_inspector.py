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
"""def create_plot(npy_filename, y_col, x_col, title):
    print("selected file is = " + str(npy_filename) + "with y_col=" + str(y_col) + " and x_col=" + str(x_col) + ", remove first="+ str(remove_first))

    #TODO if file is npy ...
    data = np.load(file=npy_filename)

    print("data.shape= " + str(data.shape))
    #print("data= " + str(data))

    print("y:" + str(y_col) + " x:" + str(x_col))

    print("data_y= " + str(data[:,y_col]))
    print("data_x= " + str(data[:,x_col]))

    plt.plot(data[:,x_col],data[:,y_col])

    plt.ylabel(COL_NAMES[y_col])
    plt.xlabel(COL_NAMES[x_col])
    plt.title(title)

    plt.savefig(filename + "_" + title + ".jpg")"""

def create_plots(npy_filename):

    # load data from file!
    data = np.load(file=npy_filename)

    # x / episode
    col_x = 1
    x_episode = data[:,col_x]

    # y1 / best total reward
    col_btr = 2
    y_btr = data[:,col_btr]

    # y2 / total reward (episode
    col_tr = 3
    y_tr = data[:,col_tr]

    # epsilon
    col_eps = 4
    y_eps = data[:,col_eps]


    plt.plot(x_episode, y_btr,'r--', x_episode, y_tr,'g.')

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("LALALA TODO ")

    plt.savefig(filename + "_" + "FANCYPLOT" + ".jpg")



# for manually creating plots....
if __name__ == "__main__":
    #remove_first=False
    # open a file dialog to pick file!!!
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()

    create_plots(npy_filename=filename)
