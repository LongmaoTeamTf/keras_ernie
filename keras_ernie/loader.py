'''
@Description: 
@version: 
@Company: Thefair
@Author: Wang Yao
@Date: 2020-03-12 19:00:08
@LastEditors: Wang Yao
@LastEditTime: 2020-03-15 18:09:21
'''
import os
import tensorflow as tf
from .convert import check_exists
from .convert import convert_paddle_to_tensor
from keras_bert import load_trained_model_from_checkpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ErnieArgs(object):
    
    def __init__(self, init_checkpoint, ernie_config_path, ernie_vocab_path, 
            max_seq_len=128, num_labels=2, use_fp16=False, use_gpu=True, gpu_memory_growth=False):
        
        check_exists(init_checkpoint)
        check_exists(ernie_config_path)
        check_exists(ernie_vocab_path)

        self.init_checkpoint = init_checkpoint
        self.ernie_config_path = ernie_config_path
        self.ernie_vocab_path = ernie_vocab_path
        
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        self.use_fp16 = use_fp16
        self.use_gpu = use_gpu
        self.gpu_memory_growth = gpu_memory_growth

        if not self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        if self.gpu_memory_growth:
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    
def load_from_checkpoint(init_checkpoint, ernie_config_path, ernie_vocab_path, ernie_version,
            max_seq_len=128, num_labels=2, use_fp16=False, use_gpu=True, gpu_memory_growth=False,
            training=False, seq_len=None, name='ernie'):
    
    args = ErnieArgs(init_checkpoint, ernie_config_path, ernie_vocab_path,
            max_seq_len=max_seq_len, num_labels=num_labels, use_fp16=use_fp16,
            use_gpu=use_gpu, gpu_memory_growth=gpu_memory_growth)

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