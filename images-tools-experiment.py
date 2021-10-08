# pip install tensorflow==2.3.1 gym==0.18.0 keras-rl2 gym[atari]
# see also https://github.com/openai/atari-py#roms
# see also http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
# see also https://punndeeplearningblog.com/development/tensorflow-cuda-cudnn-compatibility/ (for gpu support)

# also pip install pydot
# and conda install graphviz

# The following disables training on the GPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np

def load_data():

    # Load white image
    white_image = tf.keras.preprocessing.image.load_img("D:/tree/image-tools/Input_Images/image_0.png")
    white_image_array = tf.keras.preprocessing.image.img_to_array(white_image)

    # Convert images to 0, 1 scale
    # Convert tool parameters to 0, 1 scale (?)
    tf.keras.utils.image_dataset_from_directory(
        directory="D:/tree/image-tools/Output_Images",
        labels=None,
        color_mode='rgb',
        seed=1234,
        image_size=(128,128),
        validation_split=0.2,
        subset='training')

    # Data is (index, r, g, b, width, x_start, y_start, x_end, y_end)
    param_data = np.loadtxt("D:/tree/image-tools/Image_Parameters/params.txt")

    # Remove index column
    param_data = param_data[:,1:]

    # Normalize data
    maxColorChannel = 255
    maxCoordinate = 128
    minPenWidth = 1
    maxPenWidth = 20

    param_data[:,:3] /= maxColorChannel
    param_data[:,3] -= minPenWidth
    param_data[:,3] /= maxPenWidth
    param_data[:,4:] /= maxCoordinate

    for i in range(8) : 
        print(np.max(param_data[:,i]))

    print(param_data.shape)

    #tf.data.Dataset.from_tensor_slices

    return

def build_model():
    image_size = 128
    num_tool_parameters = 8

    a0 = layers.Input(shape=(image_size, image_size, 3))
    b0 = layers.Input(shape=(num_tool_parameters))
    c0 = layers.Input(shape=(image_size, image_size, 3))

    a1_head = layers.Rescaling(1./255)
    a10 = layers.Conv2D(32, kernel_size=(3, 3), activation = "relu")
    a11 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
    a12 = layers.MaxPooling2D(pool_size=(2, 2))
    a13 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
    a14 = layers.MaxPooling2D(pool_size=(2, 2))
    a15 = layers.Flatten()

    a1 = a1_head(a0)
    a1 = a10(a1)
    a1 = a11(a1)
    a1 = a12(a1)
    a1 = a13(a1)
    a1 = a14(a1)
    a1 = a15(a1)

    a2 = layers.Dense(128)(a1)
    a2 = layers.Dense(128)(a2)
    a2 = layers.Dense(128)(a2)

    b1 = layers.Dense(16)(b0)
    b1 = layers.Dense(16)(b1)
    b1 = layers.Dense(16)(b1)
    b1 = layers.Dense(16)(b1)

    b2 = layers.Dense(32)(layers.concatenate(inputs = [b1, a2]))
    b2 = layers.Dense(7*7*128)(b2)
    b2 = layers.Reshape((7,7,128))(b2)

    b3 = layers.Conv2DTranspose(64,kernel_size=(3,3), activation="relu")(b2)
    b3 = layers.MaxPooling2D(pool_size=(2,2))(b3)
    b3 = layers.Conv2DTranspose(32, kernel_size=(3,3), activation="relu")(b3)
    b3 = layers.MaxPooling2D(pool_size=(2,2))(b3)
    b3 = layers.Conv2DTranspose(32, kernel_size=(3,3), activation="relu")(b3)
    b3 = layers.Conv2DTranspose(1, kernel_size=(3,3), activation="sigmoid")(b3)
    b3 = layers.Flatten()(b3)
    b3 = layers.Dense(128*128*3)(b3)
    b3 = layers.Reshape((128,128,3))(b3)

    # Copy of a1 but taking c0 as input
    c1 = a1_head(c0)
    c1 = a10(c1)
    c1 = a11(c1)
    c1 = a12(c1)
    c1 = a13(c1)
    c1 = a14(c1)
    c1 = a15(c1)

    c2 = layers.Dense(32)(layers.concatenate(inputs = [a2, c1]))
    c2 = layers.Dense(32)(c2)

    c3 = layers.Dense(16)(c2)
    c3 = layers.Dense(16)(c3)
    c3 = layers.Dense(16)(c3)
    c3 = layers.Dense(num_tool_parameters)(c3)

    model = Model(inputs=[b0, a0, c0], outputs=[b3, c3])
    print(model.summary())

    tf.keras.utils.plot_model(model,to_file='model.png', show_shapes=True)

    return

def train_model():



    # Save weights
    return

def visualize_results():
    return

if __name__ == "__main__" : 

    load_data()
    build_model()
    train_model()
    visualize_results()
    
