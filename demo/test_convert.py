'''
@Description: 
@version: 
@Company: Thefair
@Author: Wang Yao
@Date: 2020-03-15 18:26:33
@LastEditors: Wang Yao
@LastEditTime: 2020-03-15 18:27:00
'''
import os
from keras_ernie import ErnieArgs
from keras_ernie import convert_paddle_to_tensor

ernie_path = "/root/ERNIE_stable-1.0.1"
init_checkpoint = os.path.join(ernie_path, 'params')
ernie_config_path = os.path.join(ernie_path, 'ernie_config.json')
ernie_vocab_path = os.path.join(ernie_path, 'vocab.txt')
tensor_checkpoints_dir = "/root/checkpoints"

args = ErnieArgs(init_checkpoint, ernie_config_path, ernie_vocab_path,
        max_seq_len=128, num_labels=2, use_fp16=False,
        use_gpu=True, gpu_memory_growth=False)

convert_paddle_to_tensor(args, tensor_checkpoints_dir)