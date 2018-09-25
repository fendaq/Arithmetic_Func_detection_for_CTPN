import os,shutil
import numpy as np
import tqdm

validation_data_num = 22

img_dir = "/home/tony/ocr/ocr_dataset/tal_detec_data_v2/img"
xml_dir = "/home/tony/ocr/ocr_dataset/tal_detec_data_v2/xml"

train_img_dir = "/home/tony/ocr/ocr_dataset/ctpn/train_data/img"
train_xml_dir = "/home/tony/ocr/ocr_dataset/ctpn/train_data/xml"

val_img_dir = "/home/tony/ocr/ocr_dataset/ctpn/val_data/img"
val_xml_dir = "/home/tony/ocr/ocr_dataset/ctpn/val_data/xml"

img_type = ["jpg", "png", "JPG"]

def divide_dataset():
    img_name_list = os.listdir(img_dir)
    val_data_index = []
    while len(val_data_index) != validation_data_num:
        value = np.random.randint(0, len(img_name_list)-1)
        if value not in val_data_index:
            val_data_index.append(value)

    if not os.path.exists(train_img_dir):
        os.mkdir(train_img_dir)

    if not os.path.exists(train_xml_dir):
        os.mkdir(train_xml_dir)

    if not os.path.exists(val_img_dir):
        os.mkdir(val_img_dir)

    if not os.path.exists(val_xml_dir):
        os.mkdir(val_xml_dir)

    for index in tqdm.tqdm(range(len(img_name_list))):
        img_name, img_type = img_name_list[index].split('.')
        if img_type not in img_type:
            assert 0, '{}not a img'.format(img_name_list[index])

        xml_name = img_name + '.xml'
        assert os.path.exists(os.path.join(xml_dir,xml_name)), "{} not exist".format(xml_name)

        if index in val_data_index:
            shutil.copyfile(os.path.join(img_dir, img_name_list[index]),
                            os.path.join(val_img_dir,img_name_list[index]))

            shutil.copyfile(os.path.join(xml_dir, xml_name),
                            os.path.join(val_xml_dir, xml_name))
        else:
            shutil.copyfile(os.path.join(img_dir, img_name_list[index]),
                            os.path.join(train_img_dir, img_name_list[index]))

            shutil.copyfile(os.path.join(xml_dir, xml_name),
                            os.path.join(train_xml_dir, xml_name))


    print(val_data_index)

if __name__ == "__main__":
    divide_dataset()