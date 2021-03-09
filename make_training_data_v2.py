import random
import parmap
import multiprocessing
import itertools
import os
import glob
import cv2 as cv
import numpy as np
import time

def is_building(input, segment):
    input_data = np.empty((0, 16, 16, 3))
    output_data = np.empty((0, 1))
    building_out = np.array([1])
    building_out = np.reshape(building_out, (1, 1))
    ground_out = np.array([0])
    ground_out = np.reshape(ground_out, (1, 1))
    for x in range(16):
        for y in range(16):
            input_crop = input[16*x : 16*x+16, 16*y : 16*y+16]
            #input_crop = np.array(input_crop)
            input_crop = np.reshape(input_crop, (1, 16, 16, 3))
            segment_crop = segment[16*x : 16*x+16, 16*y : 16*y+16]
            #if np.count_nonzero(segment_crop == 255) >= 128:
            if np.mean(segment_crop) >= 128:
                input_data = np.append(input_data, input_crop, axis=0)
                output_data = np.append(output_data, building_out, axis=0)
            else:
                input_data = np.append(input_data, input_crop, axis=0)
                output_data = np.append(output_data, ground_out, axis=0)
    #print(output_data)
            #print(x, y, np.shape(input_crop))
    train_data = list(zip(input_data, output_data))
    return train_data

def get_data(input_dir):
    multiprocessing.freeze_support()
    start_time = time.time()
    input = [cv.imread(file) for file in glob.glob(input_dir + 'image\\*.jpeg')]
    segment = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.glob(input_dir + 'ground\\*.png')]
    num_cores = multiprocessing.cpu_count()
    train_data = parmap.starmap(is_building, list(zip(input, segment)), pm_pbar=True, pm_processes=12)
    train_data = np.reshape(train_data, (-1, 2))
    input_data = list(zip(*train_data))[0]
    output_data = list(zip(*train_data))[1]
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    print(input_data.shape, output_data.shape)
    print("--- %s seconds --" % (time.time() - start_time))
    return input_data, output_data
if __name__ == '__main__':
    get_data()
    
    

