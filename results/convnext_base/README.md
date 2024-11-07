## train
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

## result
- it takes about 36 hours using `8 x RTX 3090`

|metric|this repository|original paper|
|---|---|---|
|top1_acc|83.54|83.82|

![image](loss_curve.png)