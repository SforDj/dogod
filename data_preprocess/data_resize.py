import skimage.io
import skimage.transform
import os
from data_preprocess.constant import *

background_img = skimage.io.imread(background_img)


def resize_and_save(size):
    if not os.path.exists(resized_new_train_img_dir):
        os.mkdir(resized_new_train_img_dir)
    if not os.path.exists(resized_new_valid_img_dir):
        os.mkdir(resized_new_valid_img_dir)

    for i in range(100):
        if not os.path.exists(resized_new_train_img_dir + "\\" + str(i)):
            os.mkdir(resized_new_train_img_dir + "\\" + str(i))
        if not os.path.exists(resized_new_valid_img_dir + "\\" + str(i)):
            os.mkdir(resized_new_valid_img_dir + "\\" + str(i))

    # for i in range(100):
    #     print(i)
    #     for f in os.listdir(new_train_img_dir + "\\" + str(i)):
    #         img_path = new_train_img_dir + "\\" + str(i) + "\\" + f
    #         img = skimage.io.imread(img_path)
    #         short_edge = min(img.shape[:2])
    #         yy = int((img.shape[0] - short_edge) / 2)
    #         xx = int((img.shape[1] - short_edge) / 2)
    #         crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    #         resized_img = skimage.transform.resize(crop_img, (size, size))
    #         resized_img_name = resized_new_train_img_dir + "\\" + str(i) + "\\" + f
    #         skimage.io.imsave(resized_img_name, resized_img)
    #
    # for i in range(100):
    #     print(i)
    #     for f in os.listdir(new_valid_img_dir + "\\" + str(i)):
    #         img_path = new_valid_img_dir + "\\" + str(i) + "\\" + f
    #         img = skimage.io.imread(img_path)
    #         short_edge = min(img.shape[:2])
    #         yy = int((img.shape[0] - short_edge) / 2)
    #         xx = int((img.shape[1] - short_edge) / 2)
    #         crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    #         resized_img = skimage.transform.resize(crop_img, (size, size))
    #         resized_img_name = resized_new_valid_img_dir + "\\" + str(i) + "\\" + f
    #         skimage.io.imsave(resized_img_name, resized_img)

    # for i in range(100):
    #     print(i)
    #     for f in os.listdir(new_train_img_dir + "\\" + str(i)):
    #         img_path = new_train_img_dir + "\\" + str(i) + "\\" + f
    #         img = skimage.io.imread(img_path)
    #         short_edge = min(img.shape[:2])
    #         long_edge = max(img.shape[:2])
    #         height = img.shape[0]
    #         width = img.shape[1]
    #
    #         resized_bg_img = skimage.transform.resize(background_img, (long_edge, long_edge))
    #         img = skimage.transform.resize(img, (height, width))
    #
    #         if height > width:
    #             margin = (long_edge - short_edge) // 2
    #             resized_bg_img[:, margin: margin + width] = img[:, :]
    #         else:
    #             margin = (long_edge - short_edge) // 2
    #             resized_bg_img[margin: margin + height, :] = img[:, :]
    #
    #         resized_img = skimage.transform.resize(resized_bg_img, (size, size))
    #         resized_img_name = resized_new_train_img_dir + "\\" + str(i) + "\\" + f
    #         skimage.io.imsave(resized_img_name, resized_img)
    #
    # for i in range(100):
    #     print(i)
    #     for f in os.listdir(new_valid_img_dir + "\\" + str(i)):
    #         img_path = new_valid_img_dir + "\\" + str(i) + "\\" + f
    #         img = skimage.io.imread(img_path)
    #         short_edge = min(img.shape[:2])
    #         long_edge = max(img.shape[:2])
    #         height = img.shape[0]
    #         width = img.shape[1]
    #
    #         resized_bg_img = skimage.transform.resize(background_img, (long_edge, long_edge))
    #         img = skimage.transform.resize(img, (height, width))
    #
    #         if height > width:
    #             margin = (long_edge - short_edge) // 2
    #             resized_bg_img[:, margin: margin + width] = img[:, :]
    #         else:
    #             margin = (long_edge - short_edge) // 2
    #             resized_bg_img[margin: margin + height, :] = img[:, :]
    #
    #         resized_img = skimage.transform.resize(resized_bg_img, (size, size))
    #
    #         resized_img_name = resized_new_valid_img_dir + "\\" + str(i) + "\\" + f
    #         skimage.io.imsave(resized_img_name, resized_img)

    for i in range(100):
        print(i)
        for f in os.listdir(new_train_img_dir + "\\" + str(i)):
            img_path = new_train_img_dir + "\\" + str(i) + "\\" + f
            img = skimage.io.imread(img_path)
            crop_img = img
            resized_img = skimage.transform.resize(crop_img, (size, size))
            resized_img_name = resized_new_train_img_dir + "\\" + str(i) + "\\" + f
            skimage.io.imsave(resized_img_name, resized_img)

    for i in range(100):
        print(i)
        for f in os.listdir(new_valid_img_dir + "\\" + str(i)):
            img_path = new_valid_img_dir + "\\" + str(i) + "\\" + f
            img = skimage.io.imread(img_path)
            crop_img = img
            resized_img = skimage.transform.resize(crop_img, (size, size))
            resized_img_name = resized_new_valid_img_dir + "\\" + str(i) + "\\" + f
            skimage.io.imsave(resized_img_name, resized_img)

resize_and_save(227)