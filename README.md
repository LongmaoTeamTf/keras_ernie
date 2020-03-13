# keras ERNIE

Pre-trained ERNIE models could be loaded for feature extraction and prediction.

## Install
```bash
pip install keras-ernie
```

## Usage

* [Download pre-trained ERNIE models](#Download-Pre-trained-ERNIE-Models)
* [Load the pre-trained ERNIE models](#Load-Pre-trained-ERNIE-Models)
* [Convert pre-trained ERNIE model to Tensor model](#Convert-Pre-trained-ERNIE-Model-To-Tensor-Model)

### Download Pre-trained ERNIE Models

Notes: Currently, only the following models are supported.

| Model                                              | Description                                                 |
| :------------------------------------------------- | :----------------------------------------------------------- |
| [ERNIE 1.0 Base for Chinese](https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz)       | with params, config and vocabs|
| [ERNIE 1.0 Base for Chinese(max-len-512)](https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz)    | with params, config and vocabs|
| [ERNIE 2.0 Base for English](https://ernie.bj.bcebos.com/ERNIE_Base_en_stable-2.0.0.tar.gz)   | with params, config and vocabs |


### Load Pre-trained ERNIE Models

```python
import os
from keras_ernie import load_from_checkpoint

ernie_path = "/root/ERNIE_stable-1.0.1"
init_checkpoint = os.path.join(ernie_path, 'params')
ernie_config_path = os.path.join(ernie_path, 'ernie_config.json')
ernie_vocab_path = os.path.join(ernie_path, 'vocab.txt')
ernie_version = "stable-1.0.1"

model = load_from_checkpoint(init_checkpoint, ernie_config_path, ernie_vocab_path, ernie_version,
            max_seq_len=128, num_labels=2, use_fp16=False, use_gpu=True, gpu_memory_growth=False,
            training=False, seq_len=None, name='ernie')
model.summary()
```

### Convert Pre-trained ERNIE Model To Tensor Model
```bash
python paddle_to_tensor.py \
    --init_checkpoint ${MODEL_PATH}/params \
    --ernie_config_path ${MODEL_PATH}/ernie_config.json \
    --ernie_vocab_path ${MODEL_PATH}/vocab.txt \
    --ernie_version stable-1.0.1 \
    --max_seq_len 128 \
    --num_labels 2 \
    --use_fp16 false \
    --use_gpu true \
    --gpu_memory_growth false \
```