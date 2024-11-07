# download
after creating a huggingface account, you need to generate your account's access token \
then, you can download datasets using the following process

```
apt-get install git-lfs
git lfs install

git clone https://huggingface.co/datasets/ILSVRC/imagenet-1k

username : your huggingface username
password : your huggingface access token
```

after the download is complete, please make the directory structure as follows
```
data/
├── train_images_0.tar.gz
├── train_images_1.tar.gz
├── train_images_2.tar.gz
├── train_images_3.tar.gz
├── train_images_4.tar.gz
└── val_images.tar.gz
```