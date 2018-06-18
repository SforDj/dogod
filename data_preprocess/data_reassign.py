import os
import shutil
from data_preprocess.constant import *


def read_files():
    with open(label_train_file, 'r', encoding='UTF-8') as r:
        train_img_label_dict = {line.split(" ")[0]: line.split(" ")[1] for line in r}
    with open(label_valid_file, 'r', encoding='UTF-8') as r:
        valid_img_label_dict = {line.split(" ")[0]: line.split(" ")[1] for line in r}

    labels = []
    for k, v in train_img_label_dict.items():
        if v not in labels:
            labels.append(v)

    label_cate_dict = {v: k for k, v in enumerate(labels)}

    if not os.path.exists(new_train_img_dir):
        os.mkdir(new_train_img_dir)

    if not os.path.exists(new_valid_img_dir):
        os.mkdir(new_valid_img_dir)

    for i in range(100):
        if not os.path.exists(new_train_img_dir + "\\" + str(i)):
            os.mkdir(new_train_img_dir + "\\" + str(i))
        if not os.path.exists(new_valid_img_dir + "\\" + str(i)):
            os.mkdir(new_valid_img_dir + "\\" + str(i))

    for f in os.listdir(raw_train_img_dir):
        img_name = f.split('.')[0]
        if img_name not in train_img_label_dict.keys():
            continue
        label = train_img_label_dict[img_name]
        cate = label_cate_dict[label]
        raw_name = raw_train_img_dir + "\\" + f
        new_name = new_train_img_dir + "\\" + str(cate) + "\\" + f
        if not os.path.exists(new_name):
            shutil.copyfile(raw_name, new_name)

    for f in os.listdir(raw_valid_img_dir):
        img_name = f.split('.')[0]
        if img_name not in valid_img_label_dict.keys():
            continue
        label = valid_img_label_dict[img_name]
        cate = label_cate_dict[label]
        raw_name = raw_valid_img_dir + "\\" + f
        new_name = new_valid_img_dir + "\\" + str(cate) + "\\" + f
        if not os.path.exists(new_name):
            shutil.copyfile(raw_name, new_name)


if __name__ == '__main__':
    read_files()