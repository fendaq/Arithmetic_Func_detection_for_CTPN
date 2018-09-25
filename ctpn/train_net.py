import os.path
import pprint
import sys
import cv2

sys.path.append(os.getcwd())
from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg

if __name__ == '__main__':
    cfg_from_file('ctpn/text.yml')
    print('Using config:')
    pprint.pprint(cfg)
    # image database
    imdb = get_imdb('voc_2007_trainval')
    # imdb.
    # print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    # print('roidb max_classes', len(roidb[0]['max_classes']))
    # print('roidb max_overlaps', len(roidb[0]['max_overlaps']))

    output_dir = get_output_dir(imdb, None)
    log_dir = get_log_dir(imdb)
    # print('Output will be saved to `{:s}`'.format(output_dir))
    # print('Logs will be saved to `{:s}`'.format(log_dir))

    device_name = '/gpu:0'
    # print(device_name)

    # img = cv2.imread(roidb[0]['image'])
    # print(roidb[0]['image'])
    # for box_index in range(len(roidb[0]['boxes'])):
    #     # print(box[0],box[1],box[2],box[3])gt_classes
    #     if roidb[0]['gt_classes'][box_index] == 1:
    #         color = (255, 0, 0)
    #     else:
    #         color = (0, 255, 0)
    #     cv2.rectangle(img, (roidb[0]['boxes'][box_index][0],roidb[0]['boxes'][box_index][1]),
    #                        (roidb[0]['boxes'][box_index][2],roidb[0]['boxes'][box_index][3]),color)
    #
    # cv2.imshow('w', img)
    # cv2.waitKey()
    # assert 0,'dwa'

    network = get_network('VGGnet_train')

    train_net(network, imdb, roidb,
              output_dir=output_dir,
              log_dir=log_dir,
              pretrained_model='data/pretrain/VGG_imagenet.npy',
              max_iters=int(cfg.TRAIN.max_steps),
              restore=bool(int(cfg.TRAIN.restore)))
