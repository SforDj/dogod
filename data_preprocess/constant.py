base_dir = u"F:\\Dog identification\\"
label_train_file = base_dir + "label_train.txt"
label_valid_file = base_dir + "label_val.txt"
raw_train_img_dir = base_dir + "train\\train"
raw_valid_img_dir = base_dir + "val\\test1"

new_train_img_dir = base_dir + "train\\new_train"
new_valid_img_dir = base_dir + "val\\new_valid"

resized_new_train_img_dir = base_dir + "train\\resized_train_3"
resized_new_valid_img_dir = base_dir + "val\\resized_valid_3"

background_img = base_dir + "background.jpg"

tfrecords_dir_white_complement = u"D:\\Github\dogod\\tfrecords\\white_complement\\"
tfrecords_dir_raw_scale = u"D:\\Github\dogod\\tfrecords\\raw_scale\\"
tfrecords_dir_cut_mid = u"D:\\Github\dogod\\tfrecords\\cut_mid\\"