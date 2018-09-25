import os
import numpy as np
import math
import cv2 as cv
from lib.prepare_training_data.parse_tal_xml import ParseXml

img_dir = '/home/tony/ocr/ocr_dataset/redo-result/img'
xml_dir = '/home/tony/ocr/ocr_dataset/redo-result/xml'

label_temp_dir = 'redo_label_tmp'
# res_path = '/home/tony/ocr/ocr_dataset/tal_detec_data_v2/xml/'

out_path = 'redo_image'
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

    for i in range(len(bbox_list)):

        if len(bbox_list[i]) == 8:
            xmin = int(np.floor(float(min(bbox_list[i][0], bbox_list[i][2], bbox_list[i][4], bbox_list[i][6])) / img_size[0] * re_size[0]))
            ymin = int(np.floor(float(min(bbox_list[i][1], bbox_list[i][3], bbox_list[i][5], bbox_list[i][7])) / img_size[1] * re_size[1]))
            xmax = int(np.ceil(float(max(bbox_list[i][0], bbox_list[i][2], bbox_list[i][4], bbox_list[i][6])) / img_size[0] * re_size[0]))
            ymax = int(np.ceil(float(max(bbox_list[i][1], bbox_list[i][3], bbox_list[i][5], bbox_list[i][7])) / img_size[1] * re_size[1]))
        elif len(bbox_list[i])==4:
            xmin = int(np.floor(float(bbox_list[i][0])/img_size[0] * re_size[0]))
            ymin = int(np.floor(float(bbox_list[i][1])/img_size[1] * re_size[1]))
            xmax = int(np.ceil(float(bbox_list[i][2])/img_size[0] * re_size[0]))
            ymax = int(np.ceil(float(bbox_list[i][3])/img_size[1] * re_size[1]))
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
        with open(os.path.join(label_temp_dir, stem) + '.txt', 'a+') as f:
            for i in range(len(x_left)):
                if class_list[i] == 0: # 手写框
                    f.writelines(class_name[class_list[i]+1])
                elif class_list[i] == 1: # 打印框
                    f.writelines(class_name[class_list[i]+1])
                else:
                    assert 0, '不该出现其他类型的class:{}'.format(class_list[i])

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
    #             cv.rectangle(re_im, (int(x_left[i]),int(ymin)), (int(x_right[i]),int(ymax)), (255,0,0),1)
    # cv.imshow('22', re_im)
    # cv.waitKey()
    # if 'hs (3)' in img_path:
    #     cv.imshow('22', re_im)
    #     cv.waitKey()
