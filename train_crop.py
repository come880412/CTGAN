import cv2
import numpy as np
import json
import os

from numpy.core import shape_base
from torch._C import dtype

if __name__ == '__main__':
    train_img_path = '../../../dataset/Sen2_MTC/data_crop'
    tiles_list = np.array(os.listdir(train_img_path))
    number_of_tiles = len(tiles_list)
    random_num_list = np.random.choice(number_of_tiles, number_of_tiles, replace=False)

    train_index = np.array(random_num_list[:int(number_of_tiles*0.7)], dtype=int)
    val_index = np.array(random_num_list[int(number_of_tiles*0.7):int(number_of_tiles*0.8)], dtype=int)
    test_index = np.array(random_num_list[int(number_of_tiles*0.8):int(number_of_tiles)], dtype=int)

    train_tiles = tiles_list[train_index]
    val_tiles = tiles_list[val_index]
    test_tiles = tiles_list[test_index]

    train_save_txt = []
    val_save_txt = []
    test_save_txt = []
    for tile in tiles_list:
        if tile[0] != 'T':
            continue
        tile_file_path = os.path.join(train_img_path, tile)
        if tile in train_tiles:
            train_save_txt.append(tile_file_path)
        elif tile in val_tiles:
            val_save_txt.append(tile_file_path)
        elif tile in test_tiles:
            test_save_txt.append(tile_file_path)
    np.savetxt('../../../dataset/Sen2_MTC/train.txt', train_save_txt, fmt='%s')
    np.savetxt('../../../dataset/Sen2_MTC/val.txt', val_save_txt, fmt='%s')
    np.savetxt('../../../dataset/Sen2_MTC/test.txt', test_save_txt, fmt='%s')