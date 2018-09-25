import os
import numpy as np
import math
import cv2 as cv
from lib.prepare_training_data.parse_tal_xml import ParseXml

train_img_dir = "/home/tony/ocr/ocr_dataset/ctpn/train_data/img"
train_xml_dir = "/home/tony/ocr/ocr_dataset/ctpn/train_data/xml"

val_img_dir = "/home/tony/ocr/ocr_dataset/ctpn/val_data/img"
val_xml_dir = "/home/tony/ocr/ocr_dataset/ctpn/val_data/xml"

img_dir = train_img_dir
xml_dir = train_xml_dir


label_temp_dir = 'train_label_tmp'
out_path = 'train_img_tmp'

proposal_width = 16.0

class_name = ['dontcare', 'handwritten', 'print']

if not os.path.exists(out_path):
    os.makedirs(out_path)
files = os.listdir(img_dir)
files.sort()

for file in files:
    _, basename = os.path.split(file)
    if basename.lower().split('.')[-1] not in ['jpg', 'png', 'JPG']:
        continue
    stem, ext = os.path.splitext(basename)
    xml_file = os.path.join(xml_dir, stem + '.xml')
    img_path = os.path.join(img_dir, file)
    # print(img_path)

    img = cv.imread(img_path)
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)

    # 图像进行resize
    re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
    re_size = re_im.shape
    cv.imwrite(os.path.join(out_path, stem) + '.jpg', re_im)

    parser = ParseXml(xml_file)
    _, class_list, bbox_list = parser.get_bbox_class()

    assert len(class_list) == len(bbox_list), 'bbox和label不对应'

    for bbox_index in range(len(bbox_list)):

        if len(bbox_list[bbox_index]) == 8:
            xmin = int(np.floor(float(min(bbox_list[bbox_index][0], bbox_list[bbox_index][2], bbox_list[bbox_index][4], bbox_list[bbox_index][6])) / img_size[0] * re_size[0]))
            ymin = int(np.floor(float(min(bbox_list[bbox_index][1], bbox_list[bbox_index][3], bbox_list[bbox_index][5], bbox_list[bbox_index][7])) / img_size[1] * re_size[1]))
            xmax = int(np.ceil(float(max(bbox_list[bbox_index][0], bbox_list[bbox_index][2], bbox_list[bbox_index][4], bbox_list[bbox_index][6])) / img_size[0] * re_size[0]))
            ymax = int(np.ceil(float(max(bbox_list[bbox_index][1], bbox_list[bbox_index][3], bbox_list[bbox_index][5], bbox_list[bbox_index][7])) / img_size[1] * re_size[1]))
        elif len(bbox_list[bbox_index])==4:
            xmin = int(np.floor(float(bbox_list[bbox_index][0])/img_size[0] * re_size[0]))
            ymin = int(np.floor(float(bbox_list[bbox_index][1])/img_size[1] * re_size[1]))
            xmax = int(np.ceil(float(bbox_list[bbox_index][2])/img_size[0] * re_size[0]))
            ymax = int(np.ceil(float(bbox_list[bbox_index][3])/img_size[1] * re_size[1]))
        else:
            print(xml_file)
            assert 0, "{}bbox error".format(xml_file)

        if xmin < 0:
            xmin = 0
        if xmax > re_size[1] - 1:
            xmax = re_size[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > re_size[0] - 1:
            ymax = re_size[0] - 1

        width = xmax - xmin + 1
        height = ymax - ymin + 1

        # TODO proposal 宽度
        step = proposal_width
        x_left = []
        x_right = []
        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
        if x_left_start == xmin:
            x_left_start = xmin + 16
        for i in np.arange(x_left_start, xmax, 16):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + 15)
        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)

        if not os.path.exists(label_temp_dir):
            os.makedirs(label_temp_dir)

        if class_list[bbox_index] == 0:  # 手写框
            current_class = class_name[class_list[bbox_index] + 1]
            color = (255, 0, 0)
        elif class_list[bbox_index] == 1:  # 打印框
            current_class = class_name[class_list[bbox_index] + 1]
            color = (0, 255, 0)
        else:
            assert 0, '不该出现其他类型的class:{}'.format(class_list[bbox_index])

        with open(os.path.join(label_temp_dir, stem) + '.txt', 'a+') as f:
            for i in range(len(x_left)):
                f.writelines(current_class)
                f.writelines("\t")
                f.writelines(str(x_left[i]))
                f.writelines("\t")
                f.writelines(str(ymin))
                f.writelines("\t")
                f.writelines(str(x_right[i]))
                f.writelines("\t")
                f.writelines(str(ymax))
                f.writelines("\n")

                # if 'hs (3)' in img_path:
                    #print((x_left[i], ymin), (x_right[i], ymax))
                    # print(str(x_left[i]), str(ymin), str(x_right[i]), str(ymax))
    #             cv.rectangle(re_im, (int(x_left[i]),int(ymin)), (int(x_right[i]),int(ymax)), color,1)
    # cv.imshow('22', re_im)
    # cv.waitKey()
    # if 'hs (3)' in img_path:
    #     cv.imshow('22', re_im)
    #     cv.waitKey()
