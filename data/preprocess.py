import os
import subprocess

if __name__ == "__main__":
    if not os.path.isdir('data/val/'):
        os.mkdir('data/val/')
        
        subprocess.run(['tar', '-xvzf', 'data/val_images.tar.gz', '-C', 'data/val/'], check = True)
        
    if not os.path.isdir('data/train/'):
        os.mkdir('data/train/')
        
        for i in range(5):
            os.mkdir(f'data/train/train_{i}/')
            
            subprocess.run(['tar', '-xvzf', f'data/train_images_{i}.tar.gz', '-C', f'data/train/train_{i}/'], check = True)
            