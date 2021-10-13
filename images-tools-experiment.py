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
import sys
import glob

from tensorflow.python.ops.gen_batch_ops import batch

def load_data():

    # Convert images to 0, 1 scale
    # Convert tool parameters to 0, 1 scale (?)

    filelist = glob.glob("D:/tree/image-tools/Output_Images/*.png")
    output_images = np.array([np.array(tf.keras.utils.load_img(fname)) for fname in filelist]).astype("float")
    output_images *= 1.0/255.

    # Load white image
    white_image_array = np.ones(output_images.shape)

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

    # Return [input, output]
    #  inputs = [ tool_param_input, current_image_input, goal_image_input ],
    return [[param_data, white_image_array, output_images],[output_images,param_data]]

def loss_function(y_true, y_pred) :
    #true_images, true_tool_params = y_true
    #predicted_images, predicted_tool_params = y_pred
    loss_object = tf.keras.losses.MeanSquaredError()
    return loss_object(y_true[0], y_pred[0]) + loss_object(y_true[1], y_pred[1])

def build_model():
    image_size = 128
    num_tool_parameters = 8

    # Inputs  : Goal image is A, Tool Params are B, and Current Image is C
    # Outputs : (B,A) -> C, (A,C) -> B
    # (Tool Params + Current Image) -> (Goal Image)
    # (Current Image + Goal Image) -> (Tool Params)
    current_image_input = layers.Input(shape=(image_size, image_size, 3))
    tool_param_input = layers.Input(shape=(num_tool_parameters))
    goal_image_input = layers.Input(shape=(image_size, image_size, 3))

    a1_head = layers.Conv2D(32, kernel_size=(3, 3), activation = "relu")
    a11 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
    a12 = layers.MaxPooling2D(pool_size=(2, 2))
    a13 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
    a14 = layers.MaxPooling2D(pool_size=(2, 2))
    a15 = layers.Flatten()

    a1 = a1_head(current_image_input)
    a1 = a11(a1)
    a1 = a12(a1)
    a1 = a13(a1)
    a1 = a14(a1)
    a1 = a15(a1)

    a2 = layers.Dense(128)(a1)
    a2 = layers.Dense(128)(a2)
    a2 = layers.Dense(128)(a2)

    b1 = layers.Dense(16)(tool_param_input)
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
    c1 = a1_head(goal_image_input)
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

    model = Model(
        inputs = [ tool_param_input, current_image_input, goal_image_input ], 
        outputs = [ b3, c3 ])
    print(model.summary())

    tf.keras.utils.plot_model(model,to_file='model.png', show_shapes=True)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss = loss_function
        )

    return model

def train_model(model, input_data, output_data):
    model.fit(x=input_data, y=output_data, batch_size = 8, epochs = 500)
    # Save weights
    model.save("b8_e500")
    return

def load_model() :
    return tf.keras.models.load_model("b8_e500")

def visualize_results():
    return

if __name__ == "__main__" : 

    input_data, output_data = load_data()
    model = build_model()

    batch_size = 2
    whiteImage = np.ones([batch_size,128,128,3])
    randomToolParams = np.random.rand(batch_size, 8)

    mockData = [randomToolParams, whiteImage, whiteImage]

    # Returned is [goalImage, toolParams]
    # where the first dimension of both goalImage and toolParams
    # is the batch dimension.
    # example : goalImage[0] is the first predicted image
    val = model.predict(mockData)

    train_model(model, input_data, output_data)
    visualize_results()
    
