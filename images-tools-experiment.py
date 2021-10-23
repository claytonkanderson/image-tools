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
import time
import matplotlib.pyplot as plt

class Experiment :
    def __init__(self) :
        self.Model = None
        self.NumToolParams = 8
        self.ImageSize = 128
        self.InputData = None
        self.OutputData = None

    def load_data(self):
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
        self.InputData = [param_data, white_image_array, output_images]
        self.OutputData = [output_images,param_data]

    def make_data_tool_only(self) :
        # tool params -> goal images
        self.InputData = self.InputData[0]
        self.OutputData = self.OutputData[0]

    def loss_function(y_true, y_pred) :
        #true_images, true_tool_params = y_true
        #predicted_images, predicted_tool_params = y_pred
        loss_object = tf.keras.losses.MeanSquaredError()
        return loss_object(y_true[0], y_pred[0]) + loss_object(y_true[1], y_pred[1])

    def build_model(self):
        image_size = self.ImageSize
        num_tool_parameters = self.NumToolParams

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

        self.Model = Model(
            inputs = [ tool_param_input, current_image_input, goal_image_input ], 
            outputs = [ b3, c3 ])
        print(self.Model.summary())

        tf.keras.utils.plot_model(self.Model,to_file='model.png', show_shapes=True)

        self.Model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
            loss = Experiment.loss_function
            )

    def build_tool_only_model(self) :
        image_size = self.ImageSize
        num_tool_parameters = self.NumToolParams

        tool_param_input = layers.Input(shape=(num_tool_parameters))

        b1 = layers.Dense(16)(tool_param_input)
        b1 = layers.Dense(16)(b1)
        b1 = layers.Dense(32)(b1)
        b1 = layers.Dense(64)(b1)
        b1 = layers.Dense(16*16*64)(b1)
        b1 = layers.Dropout(0.2)(b1)
        
        b1 = layers.Reshape((16,16,64))(b1)
        
        b2 = layers.Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), activation='relu')(b1)
        b2 = layers.Conv2DTranspose(32, kernel_size=(2,2), strides=(2,2), activation='relu')(b2)
        b2 = layers.Conv2DTranspose(16, kernel_size=(2,2), strides=(2,2), activation='relu')(b2)
        #b2 = layers.Conv2DTranspose(8, kernel_size=(2,2), strides=(1,1), activation='relu')(b2)
        b2 = layers.Conv2DTranspose(3, kernel_size=(2,2), strides=(1,1), activation='sigmoid')(b2)
        b2 = layers.AveragePooling2D(pool_size=(2,2), strides=(1,1))(b2)

        #b2 = layers.Conv2DTranspose(3, kernel_size=(2,2), strides=(2,2), activation='sigmoid')(b2)

        #b3 = layers.Reshape((image_size,image_size,3))(b2)

        self.Model = Model(
            inputs = tool_param_input, 
            outputs = b2 )
        print(self.Model.summary())

        tf.keras.utils.plot_model(self.Model,to_file='tool_only_model.png', show_shapes=True)

        self.Model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5), 
            loss = tf.keras.losses.MeanSquaredError()
            )
        return

    def train_model(self):
        self.Model.fit(x=self.InputData, y=self.OutputData, batch_size = 32, epochs = 4)
        self.Model.save("b8_e500")
        return

    def load_model(self) :
        self.Model = tf.keras.models.load_model("b8_e500")

    def visualize_results(self):
        figures = []
        for i in range(5) :
            testData = self.InputData[i].reshape(1,-1)
            val = self.Model.predict(testData)[0]
            figures.append(plt.figure())
            plt.imshow(val)
        plt.show()
        return

    def predict(self, data) :
        data[0] = data[0].reshape(1,8)
        data[1] = data[1].reshape(1,128,128,3)
        data[2] = data[2].reshape(1,128,128,3)
        return self.Model.predict(data)

if __name__ == "__main__" : 

    start = time.time()

    exp = Experiment()
    exp.load_data()
    exp.make_data_tool_only()
    exp.build_tool_only_model()
    exp.train_model()

    end = time.time()
    print("Finished after {0} seconds.".format(end-start))

    exp.visualize_results()

"""
val = predict(model, [input_data[0][1], input_data[1][1], input_data[2][1]])
temp = Image.fromarray(np.uint8(255*val[0][0]),'RGB')
temp.show()
"""