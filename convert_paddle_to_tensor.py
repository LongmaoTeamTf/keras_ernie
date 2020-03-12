'''
@Description: 
@version: 
@Company: Thefair
@Author: Wang Yao
@Date: 2020-03-12 15:08:24
@LastEditors: Wang Yao
@LastEditTime: 2020-03-12 18:31:37
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import joblib
import argparse
import numpy as np
import tensorflow as tf

import paddle.fluid as fluid
from model.ernie import ErnieConfig
from utils.init import init_checkpoint, init_pretraining_params
from finetune.classifier import create_model


os.environ["CUDA_VISIBLE_DEVICES"] = ""

gpus = tf.config.experimental.get_visible_devices(devices_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def check_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'{filepath} not exists.')
    return filepath


def convert_paddle_to_np(ernie_path, max_seq_len=128, num_labels=2, use_fp16=False, **options):

    ernie_path = check_exists(ernie_path)
    init_checkpoint = check_exists(os.path.join(ernie_path, 'params'))
    ernie_config_path = check_exists(os.path.join(ernie_path, 'ernie_config.json'))

    class args(object):
        init_checkpoint = init_checkpoint
        ernie_config_path = ernie_config_path
        max_seq_len =  max_seq_len
        num_labels = num_labels
        use_fp16 = use_fp16

    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    startup_prog = fluid.Program()
    test_program = fluid.Program()

    with fluid.program_guard(test_program, startup_prog):
        with fluid.unique_name.guard():
            _, _ = create_model(
                    args,
                    pyreader_name='test_reader',
                    ernie_config=ernie_config,
                    is_classify=True)
    
    exe.run(startup_prog)
    init_pretraining_params(
                    exe,   
                    args.init_checkpoint,
                    main_program=test_program,
                    #main_program=startup_prog,
                    use_fp16=args.use_fp16)

    name2params = {}
    prefix = args.init_checkpoint
    for var in startup_prog.list_vars():
        path = os.path.join(prefix, var.name)
        if os.path.exists(path):
            cur_tensor = fluid.global_scope().find_var(var.name).get_tensor()
            print(var.name, np.array(cur_tensor).shape)
            name2params[var.name] = np.array(cur_tensor)

    joblib.dump(name2params, 'params.dict')


def convert_np_to_tensor(params, training=False):
    tensor_prefix='bert'
    # Embeddings
    tensor_embed_prefix = f"{tensor_prefix}/embeddings"
    tf.Variable(tf.convert_to_tensor(params['pre_encoder_layer_norm_scale'], name=f"{tensor_embed_prefix}/LayerNorm/gamma"))
    tf.Variable(tf.convert_to_tensor(params['pre_encoder_layer_norm_bias'], name=f"{tensor_embed_prefix}/LayerNorm/beta"))
    tf.Variable(tf.convert_to_tensor(params['pos_embedding'], name=f"{tensor_embed_prefix}/position_embeddings"))
    tf.Variable(tf.convert_to_tensor(params['word_embedding'], name=f"{tensor_embed_prefix}/word_embeddings"))
    tf.Variable(tf.convert_to_tensor(params['sent_embedding'], name=f"{tensor_embed_prefix}/token_type_embeddings"))
    # Layers
    tensor_encoder_prefix = f"{tensor_prefix}/encoder/layer_"
    for x in range(12):
        fluid_prefix = f"encoder_layer_{x}"
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_post_att_layer_norm_scale"], 
            name=f"{tensor_encoder_prefix}{x}/attention/output/LayerNorm/gamma"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_post_att_layer_norm_bias"], 
            name=f"{tensor_encoder_prefix}{x}/attention/output/LayerNorm/beta"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_multi_head_att_output_fc.w_0"], 
            name=f"{tensor_encoder_prefix}{x}/attention/output/dense/kernel"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_multi_head_att_output_fc.b_0"], 
            name=f"{tensor_encoder_prefix}{x}/attention/output/dense/bias"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_multi_head_att_key_fc.w_0"], 
            name=f"{tensor_encoder_prefix}{x}/attention/self/key/kernel"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_multi_head_att_key_fc.b_0"], 
            name=f"{tensor_encoder_prefix}{x}/attention/self/key/bias"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_multi_head_att_query_fc.w_0"], 
            name=f"{tensor_encoder_prefix}{x}/attention/self/query/kernel"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_multi_head_att_query_fc.b_0"], 
            name=f"{tensor_encoder_prefix}{x}/attention/self/query/bias"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_multi_head_att_value_fc.w_0"], 
            name=f"{tensor_encoder_prefix}{x}/attention/self/value/kernel"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_multi_head_att_value_fc.b_0"], 
            name=f"{tensor_encoder_prefix}{x}/attention/self/value/bias"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_ffn_fc_0.w_0"], 
            name=f"{tensor_encoder_prefix}{x}/intermediate/dense/kernel"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}__ffn_fc_0.b_0"], 
            name=f"{tensor_encoder_prefix}{x}/intermediate/dense/bias"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_post_ffn_layer_norm_scale"], 
            name=f"{tensor_encoder_prefix}{x}/output/LayerNorm/gamma"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_post_ffn_layer_norm_bias"], 
            name=f"{tensor_encoder_prefix}{x}/output/LayerNorm/beta"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_ffn_fc_1.w_0"], 
            name=f"{tensor_encoder_prefix}{x}/output/dense/kernel"))
        tf.Variable(tf.convert_to_tensor(
            params[f"{fluid_prefix}_ffn_fc_1.b_0"], 
            name=f"{tensor_encoder_prefix}{x}/output/dense/bias"))
    # Pooler
    tensor_pooler_prefix = f"{tensor_prefix}/pooler"
    tf.Variable(tf.convert_to_tensor(params['pooled_fc.w_0'], name=f"{tensor_pooler_prefix}/dense/kernel"))
    tf.Variable(tf.convert_to_tensor(params['pooled_fc.b_0'], name=f"{tensor_pooler_prefix}/dense/bias"))

    if training:
        # Cls
        tf.Variable(tf.convert_to_tensor(params['mask_lm_out_fc.b_0'], name="cls/predictions/output_bias"))
        tf.Variable(tf.convert_to_tensor(params['mask_lm_trans_layer_norm_scale'], name="cls/predictions/transform/LayerNorm/gamma"))
        tf.Variable(tf.convert_to_tensor(params['mask_lm_trans_layer_norm_bias'], name="cls/predictions/transform/LayerNorm/beta"))
        tf.Variable(tf.convert_to_tensor(params['mask_lm_trans_fc.w_0'], name="cls/predictions/transform/dense/kernel"))
        tf.Variable(tf.convert_to_tensor(params['mask_lm_trans_fc.b_0'], name="cls/predictions/transform/dense/bias"))
        tf.Variable(tf.convert_to_tensor(params['next_sent_fc.w_0'], name="cls/seq_relationship/output_weights"))
        tf.Variable(tf.convert_to_tensor(params['next_sent_fc.b_0'], name="cls/seq_relationship/output_bias"))
        tf.Variable(tf.convert_to_tensor(params['cls_squad_out_w'], name="cls/squad/output_weights"))
        tf.Variable(tf.convert_to_tensor(params['cls_squad_out_b'], name="cls/squad/output_bias"))


def save_tensor(paddle_params_np):
    params = joblib.load(paddle_params_np)
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            convert_np_to_tensor(params)
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            with sess.as_default():
                checkpoint_dir = 'checkpoints'
                checkpoint_prefix = os.path.join(checkpoint_dir, 'bert_model.ckpt')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver.save(sess, checkpoint_prefix)



if __name__ == "__main__":
    ernie_path = '/media/xddz/xddz/data/ERNIE_stable-1.0.1'
    convert_paddle_to_np(ernie_path)
    save_tensor('params.dict')