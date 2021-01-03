import os
import shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def setup_file_subsystem():
    # initial file system setup
    long = False
    original_dataset_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN_Practice/Data'
    base_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN_Practice/Reduced_Data'
    if not(os.path.isdir(original_dataset_dir)):
        long = True
        original_dataset_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN-Practice/Image_Classification_CNN_Practice/Data'
        base_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN-Practice/Image_Classification_CNN_Practice/Reduced_Data'
        if not(os.path.isdir(original_dataset_dir)):
            print("Please download dataset from https://www.kaggle.com/c/dogs-vs-cats/data")
            print("Extract into " + original_dataset_dir)
            print("Then have only the images from the train folder in the directory.")
            exit(0)
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
        src = original_dataset_dir + '/' + fname
        dst = train_cats_dir + '/' + fname
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = original_dataset_dir + '/' + fname
        dst = validation_cats_dir + '/' + fname
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = original_dataset_dir + '/' + fname
        dst = test_cats_dir + '/' + fname
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = original_dataset_dir + '/' + fname
        dst = train_dogs_dir + '/' + fname
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = original_dataset_dir + '/' + fname
        dst = validation_dogs_dir + '/' + fname
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = original_dataset_dir + '/' + fname
        dst = test_dogs_dir + '/' + fname
        shutil.copyfile(src, dst)

    print("Total Training Cat Images:", len(os.listdir(train_cats_dir)))
    print("Total Validation Cat Images:", len(os.listdir(validation_cats_dir)))
    print("Total Test Cat Images:", len(os.listdir(test_cats_dir)))
    print("Total Training Dog Images:", len(os.listdir(train_dogs_dir)))
    print("Total Validation Dog Images:", len(os.listdir(validation_dogs_dir)))
    print("Total Test Dog Images:", len(os.listdir(test_dogs_dir)))
    return long


def main():
    # first time setup of the resource filesystem
    long = False
    if not(os.path.isdir('C:/Users/mcbri/PycharmProjects/Image_Classification_CNN_Practice/Reduced_Data') or os.path.isdir('C:/Users/mcbri/PycharmProjects/Image_Classification_CNN-Practice/Image_Classification_CNN_Practice/Reduced_Data')):
        long = setup_file_subsystem()
    if not long:
        train_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN_Practice/Reduced_Data/train'
        validation_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN_Practice/Reduced_Data/validation'
    else:
        train_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN-Practice/Image_Classification_CNN_Practice/Reduced_Data/train'
        validation_dir = 'C:/Users/mcbri/PycharmProjects/Image_Classification_CNN-Practice/Image_Classification_CNN_Practice/Reduced_Data/validation'
    # beginning construction of the Convnet
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # model compilation
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    # data reading and formatting
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    # Model Training
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

    model.save('cats_and_dogs_small_1.h5')

    # Plotting the results
    plot_history(history)


if __name__ == '__main__':
    main()
