
import numpy as np


class results(object):

    folder = None
    data = None
    latest_file = None

    #TODO define how result matrix will look, forexample, episode, best_reward, epsilon?,
    #row_size = 5

    def __init__(self, folder):
        self.data = []
        self.folder = folder

    def add(self, row):
        self.data.append(row)
        """if(len(row) == self.row_size):
            for r in row:
                self.data.append(r)
        else:
            raise IndexError"""


    def save(self,episode):
        #res = np.reshape(self.data, newshape=(int(len(self.data)/self.row_size) , self.row_size))
        res = np.asanyarray(self.data)
        self.latest_file = str(self.folder) + "/" + "result_at_ep-"+str(episode)+".npy"
        np.save(file=self.latest_file, arr=res)
        return self.latest_file;



    def load(self,file):
        return np.load(file=file)

    def load_latest(self):
        return np.load(file=self.latest_file)