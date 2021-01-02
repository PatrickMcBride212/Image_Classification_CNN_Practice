import os
import shutil


def main():

    # initial file system setup

    original_train_dataset_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN_Practice/Data/train/train'
    original_test_dataset_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN_Practice/Data/test1/test1'
    base_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN_Practice/Reduced_Data/small_database'
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    os.mkdir(validation_cats_dir)
    os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')

    os.mkdir(test_cats_dir)
    os.mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_train_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)


if __name__ == '__main__':
    main()
