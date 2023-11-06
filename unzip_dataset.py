from glob import glob
import os
from os.path import join

def unzip(root):
    zip_files = glob(join(root, '*.zip'))

    for f in zip_files:
        os.system(f"unzip {f} -d {root}")

if __name__ == "__main__":
    root = '.'
    unzip(root)
    zip_files = glob(join(root, '*.zip'))
    folders = map(lambda x: x.replace('.zip', ''), zip_files)
    for f in folders:
       unzip(f)

    os.system(f"rm -rf *.zip")
    os.system(f'rm -rf */*.zip')

