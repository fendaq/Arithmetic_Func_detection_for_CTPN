import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tqdm


class ParseXml(object):

    def __init__(self, xml_path, rect=False):
        self.classes = []
        self.bbox = []
        self.rect = rect
        self.img_name = xml_path.split('/')[-1].replace('.xml', '')
        # print(self.img_name)
        self.res = self._read_xml(xml_path)

    def get_bbox_class(self):

        if self.res is True:
            return self.img_name, self.classes, self.bbox
        else:
            return self.img_name, None, None

    def _read_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        itmes = root.findall("outputs/object/item")

        for i in itmes:
            res = self._parse_item(i)
            if res is False:
                return False
        return True

    def _parse_item(self, item):
        class_elem = item.find('name')

        if item.find('bndbox'):
            bbox = []
            bndbox = item.find('bndbox')


            bbox.append(int(bndbox.find('xmin').text))
            bbox.append(int(bndbox.find('ymin').text))
            bbox.append(int(bndbox.find('xmax').text))
            bbox.append(int(bndbox.find('ymax').text))
            self.bbox.append(bbox)
            self.classes.append(int(class_elem.text))
            return True
        elif item.find('polygon'):
            pos = []
            polygon = item.find('polygon')
            pos.append(int(polygon.find('x1').text))
            pos.append(int(polygon.find('y1').text))

            if polygon.find('x2') is not None:
                pos.append(int(polygon.find('x2').text))
                pos.append(int(polygon.find('y2').text))
            else:
                print('img error:', self.img_name)
                print('多边形框选有问题,少点')
                return False

            pos.append(int(polygon.find('x3').text))
            pos.append(int(polygon.find('y3').text))

            if polygon.find('y4') is not None:
                pos.append(int(polygon.find('x4').text))
                pos.append(int(polygon.find('y4').text))

                if not self.rect:
                    self.bbox.append(pos)
                else:
                    bbox = []
                    bbox.append(min(pos[0],pos[2],pos[4],pos[6]))
                    bbox.append(min(pos[1], pos[3], pos[5], pos[7]))
                    bbox.append(max(pos[0], pos[2], pos[4], pos[6]))
                    bbox.append(max(pos[1], pos[3], pos[5], pos[7]))
                    self.bbox.append(bbox)
                self.classes.append(int(class_elem.text))
            else:
                print('img error:', self.img_name)
                print('多边形框选有问题,少点')
                return False

            if polygon.find('x5'):
                print('img error:', self.img_name)
                print('多边形框选有问题.多点')
                return False

            return True
        else:
            print('img error:', self.img_name)
            print('含有其他类型bbox')
            return False

def draw_bbox(img_name, class_list, bbox_list):


    img_read_name = img_name+ '.JPG'
    if not os.path.exists(os.path.join(img_path, img_read_name)):
        img_read_name = img_name + '.jpg'

    img = cv2.imread(os.path.join(img_path, img_read_name))
    # print(bbox_list)
    # cv2.imshow('2', img)
    # cv2.waitKey()


    print_color = (0, 0, 255)
    hand_color = (255, 0, 0)

    for i in range(len(class_list)):
        if class_list[i] == 0:
            color = print_color
        else:
            color = hand_color

        if len(bbox_list[i]) == 4:
            cv2.rectangle(img, (bbox_list[i][0], bbox_list[i][1]), (bbox_list[i][2], bbox_list[i][3]), color, 2)
        else:
            cv2.line(img, (bbox_list[i][0], bbox_list[i][1]),
                     (bbox_list[i][2], bbox_list[i][3]), color, 2)
            cv2.line(img, (bbox_list[i][2], bbox_list[i][3]),
                     (bbox_list[i][4], bbox_list[i][5]), color, 2)
            cv2.line(img, (bbox_list[i][4], bbox_list[i][5]),
                     (bbox_list[i][6], bbox_list[i][7]), color, 2)
            cv2.line(img, (bbox_list[i][6], bbox_list[i][7]),
                     (bbox_list[i][0], bbox_list[i][1]), color, 2)
    #print(os.path.join('/home/tony/ocr/ocr_dataset/tal_detec_data_v2/res/', img_read_name))
    cv2.imwrite(os.path.join(res_path, img_read_name), img)



if __name__ =="__main__":

    xml_name = os.listdir(xml_path)
    for i in tqdm.tqdm(range(len(xml_name))):
        name = os.path.join(xml_path, xml_name[i])
        p = ParseXml(name)
        img_name, class_list, bbox_list = p.get_bbox_class()
        if class_list is not None:
            draw_bbox(img_name, class_list, bbox_list)

    # random_scale_inds = np.random.randint(0, high=len([600]),
    #                                 size=100)
    # print(random_scale_inds)