# imagenet1k
### simple pytorch pipeline for pretraining/finetuning vision models on imagenet-1k

1. `when you want to fine-tune vision models with a different image size` (e.g., optimizing performance for a kaggle competition dataset with an image size of 64)
    - [discussion](https://www.kaggle.com/discussions/general/543196) on this use case

2. `when you want supervied pretraining on imagenet-1k`

3. `when you want self-supervised pretraining on imagenet-1k`

## data
download the imagenet-1k dataset from [huggingface dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k) and arrange the data as follows \
for more details, check `data/`

```
data/
├── train_images_0.tar.gz
├── train_images_1.tar.gz
├── train_images_2.tar.gz
├── train_images_3.tar.gz
├── train_images_4.tar.gz
└── val_images.tar.gz
```

## setup
`
pip install -r requirements.txt
`

## preprocess
unzip the imagenet-1k located in the `data/`
```
python -m data.preprocess
```

## supervised learning
classic supervised learning on imagenet-1k dataset
```
python -m train --save_dir weights \
                --model_name convnext_base \
                --n_epoch 200 \
                --batch_size 128 \
                --n_worker 8 \
                --n_device 8 \
                --precision 16-mixed \
                --strategy ddp \
                --save_frequency 5 \
                --drop_path_rate 0.5 \
                --label_smoothing 0.1 \
                --input_size 224 
```

## self-supervised learning
currently, only facebook research's [mae](https://arxiv.org/abs/2111.06377) is supported for self-supervised learning

### pretraining
```
python -m pretraining --save_dir pretrained_weights \
                      --model_name facebook/vit-mae-base \
                      --n_epoch 400 \
                      --batch_size 256 \
                      --n_worker 8 \
                      --n_device 8 \
                      --precision 16-mixed \
                      --strategy ddp \
                      --save_frequency 20 \
                      --input_size 224 \
                      --wd 0.05 \
                      --norm_pix_loss
```

### finetuning
```
python -m finetuning --save_dir weights \
                     --model_name facebook/vit-mae-base \
                     --pretrained_dir pretrained_weights \
                     --n_epoch 200 \
                     --batch_size 128 \
                     --n_worker 8 \
                     --n_device 8 \
                     --precision 16-mixed \
                     --strategy ddp \
                     --save_frequency 5 \
                     --input_size 224 \
                     --drop_path_rate 0.1 \
                     --label_smoothing 0.1 \
                     --wd 0.05 
```

## results
- check `results/` 

## acknowledgement
this project makes use of the following libraries and models
- [timm](https://github.com/huggingface/pytorch-image-models)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [vision_transformer](https://github.com/google-research/vision_transformer)
- [mae](https://github.com/facebookresearch/mae)
- [transformers](https://github.com/huggingface/transformers)