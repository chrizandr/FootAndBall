# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Run FootAndBall detector on ISSIA-CNR Soccer videos
#

import torch
import cv2
import os
import argparse
import pdb

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL

import sys
from tqdm import tqdm
import numpy as np

sys.path.insert(1, '/home/chrizandr')

from yolov3.annotations.annot.annot_utils import CVAT_Track, CVAT_annotation


def draw_bboxes(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            color = (255, 0, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (int(x1), max(0, int(y1)-10)), font, 1, color, 2)

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            color = (0, 0, 255)
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), font, 1,
                        color, 2)

    return image


def run_detector(model, args):
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    _, file_name = os.path.split(args.path)

    if args.device == 'cpu':
        print('Loading CPU weights...')
        state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
    else:
        print('Loading GPU weights...')
        state_dict = torch.load(args.weights)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    print('Processing video: {}'.format(args.path))

    images = os.listdir(args.path)
    annotation = CVAT_annotation()

    for img in tqdm(images):
        frame_img = cv2.imread(os.path.join(args.path, img))
        frame = int(img.strip("image").strip(".jpg"))
        img_tensor = augmentations.numpy2tensor(frame_img)

        track = CVAT_Track(frame)
        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)
            detections = model(img_tensor)[0]
        # if len(detections['boxes']) > 10:
        #     pdb.set_trace()
        for det, label, conf in zip(detections['boxes'].cpu(), detections['labels'].cpu(), detections['scores'].cpu()):
            xtl, ytl, xbr, ybr = np.array(det)
            if label.item() == 2 and conf.item() > 0.5:
                track.create_bbox(frame-1, xtl, ytl, xbr, ybr, conf=conf.item())
        if len(track.bboxes) != 0:
            annotation.insert_track(track)

    annotation.build(args.annot, add_conf=True)


if __name__ == '__main__':
    print('Run FootAndBall detector on input video')

    # Train the DeepBall ball detector model
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to video', type=str, required=True)
    parser.add_argument('--annot', help='path to video', type=str, required=True)
    parser.add_argument('--model', help='model name', type=str, default='fb1')
    parser.add_argument('--weights', help='path to model weights', type=str, required=True)
    parser.add_argument('--ball_threshold', help='ball confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--player_threshold', help='player confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--out_video', help='path to video with detection results', type=str, required=False,
                        default=None)
    parser.add_argument('--device', help='device (CPU or CUDA)', type=str, default='cuda:0')
    args = parser.parse_args()

    print('Video path: {}'.format(args.path))
    print('Model: {}'.format(args.model))
    print('Model weights path: {}'.format(args.weights))
    print('Ball confidence detection threshold [0..1]: {}'.format(args.ball_threshold))
    print('Player confidence detection threshold [0..1]: {}'.format(args.player_threshold))
    print('Output video path: {}'.format(args.out_video))
    print('Device: {}'.format(args.device))

    print('')

    assert os.path.exists(args.weights), 'Cannot find FootAndBall model weights: {}'.format(args.weights)
    assert os.path.exists(args.path), 'Cannot open video: {}'.format(args.path)

    model = footandball.model_factory(args.model, 'detect', ball_threshold=args.ball_threshold,
                                      player_threshold=args.player_threshold)

    run_detector(model, args)

# python run_detector.py --path /ssd_scratch/cvit/chrizandr/images --weights models/model_20201019_1416_final.pth --annot frvscr_fnb.xml --device cuda
