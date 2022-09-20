# SIHRNet
code of SIHRNet

paper link:
https://www.researchgate.net/publication/360639045_SIHRNet_a_fully_convolutional_network_for_single_image_highlight_removal_with_a_real-world_dataset

# Download the 2022 SIHR dataset:
https://drive.google.com/file/d/1njwFmTTmTwrMxazCMn15E8ebrVl0oNBN/view?usp=sharing

trained ckpt:
https://drive.google.com/file/d/1ZUE0P9mW-fjC9j5ZkKyHva8MG_V8RYqs/view?usp=sharing

VGG_model:
https://drive.google.com/file/d/1mH0VEJtQXa245OQMAHuUhi4xRPnqqs66/view?usp=sharing

# train:
python main.py --data_dir "data_direction"

# test:

python main.py --is_training 0

you should change the root of test set in the code

