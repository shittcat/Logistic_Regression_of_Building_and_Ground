from make_training_data_v2 import get_data
from Logistic_Regression import Logistic_Regression_train
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    train_path = ".\\train\\"
    test_path = ".\\test\\"
    
    train_input, train_output = get_data(train_path)
    test_input, test_output = get_data(test_path)

    Logistic_Regression_train(train_input, train_output, test_input, test_output)
