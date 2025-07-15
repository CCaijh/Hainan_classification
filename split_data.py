import os
from shutil import copy, rmtree
import random

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    random.seed(0)

    split_rate_val = 0.2
    split_rate_test = 0
    split_rate_train = 1 - (split_rate_val + split_rate_test)

    cwd = os.getcwd()
    origin_flower_path = os.path.join("\HA419-IonogramSet-202505\HA_80_20_uniform")
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    flower_class = [cla for cla in os.listdir(origin_flower_path) if os.path.isdir(os.path.join(origin_flower_path, cla))]

    train_root = os.path.join(".\Hainan_classification", "HA_80_20_uniform/train")
    mk_file(train_root)
    for cla in flower_class:
        mk_file(os.path.join(train_root, cla))

    val_root = os.path.join(".\Hainan_classification", "HA_80_20_uniform/val")
    mk_file(val_root)
    for cla in flower_class:
        mk_file(os.path.join(val_root, cla))

    test_root = os.path.join(".\Hainan_classification", "HA_80_20_uniform/test")
    mk_file(test_root)
    for cla in flower_class:
        mk_file(os.path.join(test_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        random.shuffle(images)  

        train_end = int(num * split_rate_train)
        val_end = int(num * (split_rate_train + split_rate_val))

        for index, image in enumerate(images):
            if index < train_end:
                new_path = os.path.join(train_root, cla)
            elif index < val_end:
                new_path = os.path.join(val_root, cla)
            else:
                new_path = os.path.join(test_root, cla)

            image_path = os.path.join(cla_path, image)
            copy(image_path, new_path)

            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")

        print()

    print("processing done!")

if __name__ == '__main__':
    main()
