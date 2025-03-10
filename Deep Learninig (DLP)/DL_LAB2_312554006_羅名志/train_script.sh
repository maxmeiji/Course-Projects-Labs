#!/bin/bash

python main.py --learning_rate 0.0001 --num_epochs 60 --batch_size 64 --model ResNet50 --demo 0
python main.py --learning_rate 0.0001 --num_epochs 60 --batch_size 64 --model VGG19 --demo 0