#########################################################
#                                                       #
# Fonctions utiles au projet de classification d'images #
#                                                       #
#########################################################

import os
import re
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as plticker
import matplotlib.gridspec as gridspec

import tensorflow as tf


def count_images(n_categories, base_dir):
    """
    Count number of images in the n_categories most populated categories
    """
    cat_counts = []

    for category in os.listdir(base_dir):
        if category.startswith('.'):
            continue
        cat_name = re.match(r"^n\d*-([\w\-]*)", category).groups()[0].title()
        cat_size = len(os.listdir(os.path.join(base_dir, category)))
        cat_counts.append((cat_name, cat_size))

    cat_counts = sorted(cat_counts, key=lambda x: x[1], reverse=True)
    return [cat_name for cat_name, _ in cat_counts[:n_categories]]


def organize_data(original_dir='Images', base_dir='data', n_categories=120):
    """
    Organize data into new directories as to be ready for Keras' Generators
    Separates original data into training, validation and testing sets
    """

    directories = os.listdir(original_dir)
    # filter out non-data directories
    directories = [directory for directory in directories
                   if not directory.startswith('.')]

    # Create folds to contain training, validation and testing sets
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    valid_dir = os.path.join(base_dir, 'validation')
    os.mkdir(valid_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    train_size = 0
    valid_size = 0
    test_size = 0

    of_interest = count_images(n_categories, original_dir)

    for directory in directories:
        # Extract class name of dogs from each directory name and format
        new_dir = re.match(r"^n\d*-([\w\-]*)", directory).groups()[0].title()
        if new_dir not in of_interest:
            continue
        images = os.listdir(os.path.join(original_dir, directory))

        # Separate randomly images of each class into each of the 3 sets
        np.random.seed(2019)  # for reproducibility
        np.random.shuffle(images)
        n_train = round(len(images) * 0.6)
        n_valid = round(len(images) * 0.2)
        n_test = n_train + n_valid
        os.mkdir(os.path.join(train_dir, new_dir))
        for image in images[:n_train]:
            src = os.path.join(original_dir, directory, image)
            dst = os.path.join(train_dir, new_dir, image)
            shutil.copyfile(src, dst)
            train_size += 1
        os.mkdir(os.path.join(valid_dir, new_dir))
        for image in images[n_train:n_test]:
            src = os.path.join(original_dir, directory, image)
            dst = os.path.join(valid_dir, new_dir, image)
            shutil.copyfile(src, dst)
            valid_size += 1
        os.mkdir(os.path.join(test_dir, new_dir))
        for image in images[n_test:]:
            src = os.path.join(original_dir, directory, image)
            dst = os.path.join(test_dir, new_dir, image)
            shutil.copyfile(src, dst)
            test_size += 1
        # print("DONE: ", new_dir)

    # print("\n")
    print("TRAIN SIZE: ", train_size)
    print("VALID SIZE: ", valid_size)
    print("TEST SIZE: ", test_size)


def create_generators(base_dir, batch_size, side_length,
                      augmentation=False, preprocess_input=None):
    """
    Define image generators for training, validation and testing steps
    """

    # training generator with data augmentation transformations
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess_input
    )

    # inference is not to be made on transformed images
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    target_size = (side_length, side_length)

    if augmentation:
        train_generator = train_datagen.flow_from_directory(
            directory=os.path.join(base_dir, 'train'),
            target_size=target_size,
            class_mode='categorical',
            batch_size=batch_size,
        )
    else:
        train_generator = valid_datagen.flow_from_directory(
            directory=os.path.join(base_dir, 'train'),
            target_size=target_size,
            class_mode='categorical',
            batch_size=batch_size,
        )

    valid_generator = valid_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'validation'),
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
    )

    # inference during testing one step at a time
    test_generator = valid_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'test'),
        target_size=target_size,
        class_mode='categorical',
        batch_size=1,
        shuffle=False
    )

    return train_generator, valid_generator, test_generator


def vgg16_preprocess_input(x, mode='tf'):
    "Wrapper of the initial vgg preprocessing function with tf compute mode"

    return tf.keras.applications.vgg16.preprocess_input(x, mode='tf')


def learning_curves(history, smoothed=False, factor=0.8, mult_ticks=2):
    """
    Plot learning curves of performance measures recorded during training
    measures: accuracy, loss; on training, validation data

    optional:
        - smoothing out the curve to capture the overall trend
        - if many epochs, possibility of spacing out the ticks on x-axis
    """
    def to_smooth(points, factor=factor):
        smoothed_points = []
        for point in points:
            if smoothed_points:  # if the list is not empty
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor +
                                       point * (1 - factor))
            else:  # if the list is empty
                smoothed_points.append(point)
        return smoothed_points

    acc_title = 'Training and Validation accuracy'
    loss_title = 'Training and Validation loss'

    if smoothed:
        train_acc = to_smooth(history.history['acc'])
        train_loss = to_smooth(history.history['loss'])
        val_acc = to_smooth(history.history['val_acc'])
        val_loss = to_smooth(history.history['val_loss'])
        acc_title = 'Smoothed ' + acc_title
        loss_title = 'Smoothed ' + loss_title
    else:
        train_acc = history.history['acc']
        train_loss = history.history['loss']
        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    epochs = range(1, len(train_acc) + 1)

    ax[0].plot(epochs, train_acc, 'bo', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'b-', label='Validation accuracy')
    ax[0].set_title(acc_title)
    ax[0].legend()

    ax[1].plot(epochs, train_loss, 'ro', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r-', label='Validation loss')
    ax[1].set_title(loss_title)
    ax[1].legend()
    ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=mult_ticks))

    plt.tight_layout()
    plt.show()


def report2dict(cr):
    """
    Transform the classification_report output of the sklearn library
    from string to named dictionary for each class and each performance measure
    """
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    avgs = ['weighted avg', 'macro avg', 'micro avg']
    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        if class_label.strip() not in avgs:
            for j, m in enumerate(measures):
                D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data


def make_report(true_labels, predicted_labels, target_names):
    """
    Transform the classification_report output of the sklearn library
    From string to clean data frame. One row per class, one column per metric
    """
    matrix_report = classification_report(true_labels, predicted_labels,
                                          target_names=target_names)

    # Transform the dictionary returned to a dataframe
    dict_report = report2dict(matrix_report)
    df_report = pd.DataFrame(dict_report).T

    return df_report


def generator_infos(generator):
    """
    Extracts general informations on generators from their attributes

    Returns
        - dictionary to convert name_class -> indice
        - dictionary inverse: indice -> name_class
        - paths of all files in the source directory of the generator
        - labels of all files in the source directory of the generator
    """
    dict_classes = generator.class_indices
    dict_classes_inv = {v: k for k, v in dict_classes.items()}
    filepaths = np.array(generator.filepaths)
    filelabels = generator.labels

    return dict_classes, dict_classes_inv, filepaths, filelabels


def class_infos(current_class, generator, probas, predictions):
    """
    Subselects generator and inference data for a specific class

    Returns:
        - Class indice corresponding to the class name
        - Paths of files labeled as given current_class
        - Probability outputs infered from files labeled as given current_class
        - Classes infered from files labeled as given current_class
    """

    dict_classes, dict_classes_inv, filepaths, filelabels =\
        generator_infos(generator)

    class_indice = dict_classes[current_class]  # indice of analysed class
    # subset of all images labeled as given current_class
    class_mask = filelabels == class_indice

    class_paths = filepaths[class_mask]  # paths of images of that class
    class_probas = probas[class_mask]  # probas of images of that classes
    # labels inferend for images labeled as given current_class
    class_predictions = predictions[class_mask]

    return class_indice, class_paths, class_probas, class_predictions


def class_analysis(current_class, generator, probas, predictions):
    """
    The goal of this function is to visualise the most common predicted classes
    by a given model, for a group of images of same ground truth
    """

    dict_classes, dict_classes_inv, filepaths, filelabels =\
        generator_infos(generator)

    class_indice, class_paths, class_probas, class_predictions =\
        class_infos(current_class, generator, probas, predictions)

    # See what labels are predicted, and their frequence
    uniq_val, uniq_freq = np.unique(class_predictions, return_counts=True)

    # Order by most frequency
    zip_list = sorted(zip(uniq_val, uniq_freq),
                      key=lambda x: x[1], reverse=True)
    uniq_classes = [dict_classes_inv[indice] for indice, freq in zip_list]

    # green for true class, red for errors
    color_classes = ['lime' if name == current_class else 'r'
                     for name in uniq_classes]
    uniq_freqs = [freq for indice, freq in zip_list]
    plt.bar(uniq_classes, uniq_freqs, color=color_classes)
    plt.xticks(rotation=45)
    plt.title(f"Predicted classes for {current_class} images")


def instance_analysis(current_class, generator, probas, predictions):
    """
    The goal of this function is to have a clear visualisation of where our
    model has failed the most in its inference phase.

    It will show the images where the model has infered the lowest
    probabilities on their ground truth class, for a given class
    """

    dict_classes, dict_classes_inv, filepaths, filelabels =\
        generator_infos(generator)

    class_indice, class_paths, class_probas, class_predictions =\
        class_infos(current_class, generator, probas, predictions)

    # proba output for each file to be of current_class
    specific_probas = class_probas[:, class_indice]
    n_examples = 6

    # the [n_examples] indices, name files with lowest probas infered
    lowest_probas_ind = specific_probas.argsort()[:n_examples]
    lowest_probas_files = class_paths[lowest_probas_ind]
    lowest_probas_classes = class_probas[lowest_probas_ind]

    # define the outer grid
    grid_row, grid_col = 2, 3
    inner_row, inner_col = 2, 1
    fig = plt.figure(figsize=(8, 8))
    outer = gridspec.GridSpec(grid_row, grid_col, wspace=0.2, hspace=0.2)

    for i in range(grid_row * grid_col):
        inner = gridspec.GridSpecFromSubplotSpec(inner_row, inner_col,
                                                 subplot_spec=outer[i],
                                                 wspace=0.1, hspace=0)

        # for each of the lowest probabilities infered, sort all probas append
        lowest_probas_classes_ind = lowest_probas_classes[i].argsort()[-4:]
        lowest_probas_classes_val =\
            lowest_probas_classes[i, lowest_probas_classes_ind]

        lowest_probas_classes_names = [dict_classes_inv[ind]
                                       for ind in lowest_probas_classes_ind]
        lowest_probas_classes_colors = ['palegreen' if name == current_class
                                        else 'salmon'
                                        for name in lowest_probas_classes_names]

        lowest_probas_classes_colors.append('skyblue')
        lowest_probas_classes_names.append(current_class)
        lowest_probas_classes_val = np.append(lowest_probas_classes_val, 1.0)

        for j in range(inner_row * inner_col):
            ax = plt.Subplot(fig, inner[j])

            if j == 0:
                img = mpimg.imread(lowest_probas_files[i])
                ax.imshow(img, aspect="auto")

            if j == 1:
                ax.barh(range(5), lowest_probas_classes_val, height=1,
                        color=lowest_probas_classes_colors)

                for k, name in enumerate(lowest_probas_classes_names):
                    ax.text(s=name, x=0, y=k, color="k", size=10,
                            verticalalignment="center")

            ax.axis('off')
            fig.add_subplot(ax)
