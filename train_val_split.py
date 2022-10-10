import numpy as np
import os
from utils import fixed_seed


if __name__ == '__main__':
    fixed_seed(2022)
    data_path = "../dataset"

    train_img_path = os.path.join(data_path, "Sen2_MTC", "Sen2_MTC")
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
        if tile in train_tiles:
            train_save_txt.append(tile)
        elif tile in val_tiles:
            val_save_txt.append(tile)
        elif tile in test_tiles:
            test_save_txt.append(tile)
    np.savetxt(os.path.join(data_path, "train.txt"), train_save_txt, fmt='%s')
    np.savetxt(os.path.join(data_path, "val.txt"), val_save_txt, fmt='%s')
    np.savetxt(os.path.join(data_path, "test.txt"), test_save_txt, fmt='%s')