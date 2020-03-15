'''
@Description: 
@version: 
@Company: Thefair
@Author: Wang Yao
@Date: 2020-03-12 22:46:00
@LastEditors: Wang Yao
@LastEditTime: 2020-03-13 00:51:11
'''
import os
from keras_ernie import load_from_checkpoint


ernie_path = "/media/xddz/xddz/data/ERNIE_stable-1.0.1"
init_checkpoint = os.path.join(ernie_path, 'params')
ernie_config_path = os.path.join(ernie_path, 'ernie_config.json')
ernie_vocab_path = os.path.join(ernie_path, 'vocab.txt')
ernie_version = "stable-1.0.1"

model = load_from_checkpoint(init_checkpoint, ernie_config_path, ernie_vocab_path, ernie_version,
            max_seq_len=128, num_labels=2, use_fp16=False, use_gpu=True, gpu_memory_growth=False,
            training=False, seq_len=None, name='ernie')
model.summary()