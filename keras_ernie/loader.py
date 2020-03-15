'''
@Description: 
@version: 
@Company: Thefair
@Author: Wang Yao
@Date: 2020-03-12 19:00:08
@LastEditors: Wang Yao
@LastEditTime: 2020-03-15 18:41:28
'''
import os
import tensorflow as tf
from .convert import check_exists
from .convert import convert_paddle_to_tensor
from keras_bert import load_trained_model_from_checkpoint


class ErnieArgs(object):
    
    def __init__(self, init_checkpoint, ernie_config_path, ernie_vocab_path, 
            max_seq_len=128, num_labels=2, use_fp16=False):
        
        check_exists(init_checkpoint)
        check_exists(ernie_config_path)
        check_exists(ernie_vocab_path)

        self.init_checkpoint = init_checkpoint
        self.ernie_config_path = ernie_config_path
        self.ernie_vocab_path = ernie_vocab_path
        
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        self.use_fp16 = use_fp16

    
def load_from_checkpoint(init_checkpoint, ernie_config_path, ernie_vocab_path, ernie_version,
            max_seq_len=128, num_labels=2, use_fp16=False, training=False, seq_len=None, name='ernie'):
    
    args = ErnieArgs(init_checkpoint, ernie_config_path, ernie_vocab_path,
            max_seq_len=max_seq_len, num_labels=num_labels, use_fp16=use_fp16)

    checkpoints_dir = os.path.join('tmp', f"ernie_{ernie_version}")
    if not os.path.exists(checkpoints_dir):
        convert_paddle_to_tensor(args, checkpoints_dir)
    
    bert_config_path = os.path.join(checkpoints_dir, 'bert_config.json')
    bert_checkpoint_path = os.path.join(checkpoints_dir, 'bert_model.ckpt')

    model = load_trained_model_from_checkpoint(
            bert_config_path, bert_checkpoint_path, training=training, seq_len=seq_len)
    
    model.name = name
    
    return model


if __name__ == "__main__":
    ernie_path = "/media/xddz/xddz/data/ERNIE_stable-1.0.1"
    init_checkpoint = os.path.join(ernie_path, 'params')
    ernie_config_path = os.path.join(ernie_path, 'ernie_config.json')
    ernie_vocab_path = os.path.join(ernie_path, 'vocab.txt')
    ernie_version = "stable-1.0.1"
    model = load_from_checkpoint(
        init_checkpoint, ernie_config_path, ernie_vocab_path, ernie_version,)
    model.summary()