import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenetv2 import preprocess_input

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    with open('class_traductor.pkl', 'rb') as f:
        traductor = pickle.load(f)
    # Inverse dictionary: class index -> class name
    traductor = {v: k for k, v in traductor.items()}

    generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    ).flow_from_directory(
        directory='images',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=1,
        shuffle=False
    )

    model = load_model('classifier.h5')
    predictions = model.predict_generator(generator, steps=generator.n)
    predictions = np.argmax(predictions, axis=1)

    # link_image = sys.argv[1]
    # image = imread(link_image, mode='RGB')
    # image = imresize(image, (224, 224))
    # image = np.reshape(image, (1, *image.shape))
    # image = image / 255

    print("\n\nPREDICTIONS:")
    for n, prediction in enumerate(predictions):
        print(f"\t{generator.filenames[n]}: {traductor[prediction]}")
