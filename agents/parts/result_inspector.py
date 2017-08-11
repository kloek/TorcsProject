import tkinter as tk
from tkinter import filedialog
import numpy as np
import numpy.ma as ma
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
    # [episode, self.total_steps, self.best_training_reward, self.best_testing_reward, total_reward, train_indicator, self.epsilon, early_stop, ob['damage']])

    # load data from file!
    data = np.load(file=npy_filename)

    # x / episode / steps
    #col_x = 0 # for episode as x
    col_x = 1 # for steps as x

    # best training reward
    where = np.logical_and((np.isfinite(data[:,2])),(np.equal(data[:,5],1)))
    train_x = data[where,col_x]
    train_y = data[where,2]

    # best testing reward
    where = np.logical_and((np.isfinite(data[:, 3])), (np.equal(data[:, 5], 0)))
    best_test_x = data[where, col_x]
    best_test_y = data[where, 3]

    # all testing reward
    where = np.logical_and((np.isfinite(data[:, 3])), (np.equal(data[:, 5], 0)))
    all_test_x = data[where, col_x]
    all_test_y = data[where, 4]


    # total reward ( episode )
    where = np.isfinite(data[:, 4])
    reward_x = data[where, col_x]
    reward_y = data[where, 4]

    # epsilon
    col_eps = 6
    epsilon_x = data[:, col_x]
    epsilon_y = data[:,col_eps]

    # epsilon
    col_damage = 8
    damage_x = data[:, col_x]
    damage_y = data[:,col_damage]


    # create reward plot
    #plt.plot(train_x, train_y, 'r-', test_x, test_y, 'k-', reward_x, reward_y, 'g.')
    train_line = plt.plot(train_x, train_y, 'b--', label="Max Training Reward")
    reward_dots = plt.plot(reward_x, reward_y, 'g.', label="Episode Reward")
    best_test_line = plt.plot(best_test_x, best_test_y, 'k-', label="Max Testing Reward")
    all_test_line = plt.plot(all_test_x, all_test_y, 'r--', label="All Testing Reward")
    damage_line = plt.plot(damage_x, damage_y, 'y--', label="Damage")

    # add legend
    plt.legend()

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Reward ")

    plt.savefig(npy_filename + "_" + "Reward" + ".jpg")





# for manually creating plots....
if __name__ == "__main__":
    #remove_first=False
    # open a file dialog to pick file!!!
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()

    create_plots(npy_filename=filename)
