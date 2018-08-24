#!/usr/bin/env bash
python AE_brain.py --total-data data/glove-vgg-sound-full.txt --train-data data/glove_vgg_sound_m.txt --brain-data data/ass_emb_m.txt --text-dim 300 --image-dim 128 --text-dim1 250 --text-dim2 150 --image-dim1 90 --image-dim2 60 --sound-dim 128 --sound-dim1 90 --sound-dim2 60 --multi-dim 300 --brain-dim1 350 --brain-dim 556  --batch-size 64 --epoch 600 --load-model result/ae-m300-250-150.parameters-100 --outmodel result_brain/ae_brain_m-m300 --gpu 1 > log.ae_brain2_rel_m

