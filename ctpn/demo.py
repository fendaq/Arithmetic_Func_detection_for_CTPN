from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):

    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])

    iiimg = cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    return iiimg, f


def draw_boxes(img, boxes, color):
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[5])), color, 1)


def ctpn(sess, net, image_path):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_path)
    img_name = image_path.split('/')[-1]
    #　将图像进行resize并返回其缩放大小
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # 送入网络得到1000个得分,1000个bbox
    cls, scores, boxes = test_ctpn(sess, net, img)

    print('cls, scores, boxes',cls.shape, scores.shape, boxes.shape)

    # img_re = img
    # for i in range(np.shape(boxes)[0]):
    #     if cls[i] == 1:
    #         color = (255, 0, 0)
    #     else:
    #         color = (0, 255, 0)
    #     cv2.rectangle(img_re, (boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),color,1)
    # cv2.imwrite(os.path.join('./data/proposal_res', img_name), img_re)

    handwritten_filter = np.where(cls==1)[0]
    handwritten_scores = scores[handwritten_filter]
    handwritten_boxes = boxes[handwritten_filter, :]

    print_filter = np.where(cls==2)[0]
    print_scores = scores[print_filter]
    print_boxes = boxes[print_filter, :]

    handwritten_detector = TextDetector()
    handwritten_detector = TextDetector()

    print('print_filter', np.array(print_filter).shape)
    print('handwritten_boxes, handwritten_scores', handwritten_boxes.shape, handwritten_scores[:, np.newaxis].shape)

    filted_handwritten_boxes = handwritten_detector.detect(handwritten_boxes, handwritten_scores[:, np.newaxis], img.shape[:2])
    filted_print_boxes = handwritten_detector.detect(print_boxes, print_scores[:, np.newaxis], img.shape[:2])

    # boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, filted_handwritten_boxes, (255,0,0))
    draw_boxes(img, filted_print_boxes, (0,255,0))

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", img_name), img)

    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network 构建网络模型
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    #im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*'))
    # print(glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.JPG')))
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name)
