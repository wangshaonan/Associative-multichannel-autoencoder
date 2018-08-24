#!/usr/bin/env bash

python AE.py --train-data data/glove-vgg-sound-full.txt --text-dim 300 --image-dim 128 --text-dim1 250 --text-dim2 150 --image-dim1 90 --image-dim2 60 --sound-dim 128 --sound-dim1 90 --sound-dim2 90 --multi-dim 300 --batch-size 64 --epoch 300 --outmodel result/ae-m300-250-150 --gpu 1 > log.300-250-150--128-90-60--300


