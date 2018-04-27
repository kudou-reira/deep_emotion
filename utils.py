import io
import cv2
import base64 
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

trained_model_checkpoint = './model2/emotion_model_final'
trained_model_graph = './model2/emotion_model_final.meta'

emotion = {0: 'Angry', 1: 'Fear', 2: 'Happy',
           3: 'Sad', 4: 'Surprise', 5: 'Neutral'}

def sample(img):
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return new_image_string


def evaluate(result):
    index_max = np.argmax(result[0])
    emotion_result = ''
    if index_max == 0:
        emotion_result = emotion[0]
    elif index_max == 1:
        emotion_result = emotion[1]
    elif index_max == 2:
        emotion_result = emotion[2]
    elif index_max == 3:
        emotion_result = emotion[3]
    elif index_max == 4:
        emotion_result = emotion[4]
    elif index_max == 5:
        emotion_result = emotion[5]

    # "{0:.2f}".format(result[0][index_max])
    probability = "{0:.2f}".format(result[0][index_max])
    print("this is probability", probability)

    return probability, emotion_result

def resize_flatten_image(image, square_size):
    height, width = image.shape
    if height > width:
        differ = height
    else:
        differ = width
    differ += 4

    mask = np.zeros((differ, differ), dtype="uint8")
    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)
    mask[y_pos:y_pos + height, x_pos:x_pos + width] = image[0:height, 0:width]
    mask = cv2.resize(mask, (square_size, square_size), interpolation=cv2.INTER_AREA)
    mask = mask.reshape(1, 2304)

    return mask


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    base64_string = base64_string.replace('data:image/jpeg;base64,','')
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def toGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def predict(image):
    emotion_graph = tf.Graph()
    with tf.Session(graph=emotion_graph) as sess:
        x = tf.placeholder(tf.float32, shape=[None, 2304])
        y_true = tf.placeholder(tf.float32, shape=[None, 6])

        # layers
        x_image = tf.reshape(x, [-1, 48, 48, 1])

        convo_1 = convolutional_layer(x_image, shape=[3, 3, 1, 32])
        convo_1_pooling = max_pool_2by2(convo_1)

        convo_2 = convolutional_layer(convo_1_pooling, shape=[3, 3, 32, 64])
        convo_2_pooling = max_pool_2by2(convo_2)

        convo_3 = convolutional_layer(convo_2_pooling, shape=[3, 3, 64, 128])
        convo_3_pooling = max_pool_2by2(convo_3)

        convo_4 = convolutional_layer(convo_3_pooling, shape=[3, 3, 128, 256])
        convo_4_pooling = max_pool_2by2(convo_4)

        convo_4_flat = tf.reshape(convo_4_pooling, [-1, 3 * 3 * 256])
        full_layer_1 = normal_full_layer(convo_4_flat, 4096)
        full_layer_2 = tf.nn.relu(normal_full_layer(full_layer_1, 1024))

        hold_prob = tf.placeholder(tf.float32)
        full_one_dropout = tf.nn.dropout(full_layer_2, keep_prob=hold_prob)

        y_pred = normal_full_layer(full_one_dropout, 6)
        y_prob = tf.nn.softmax(y_pred)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)

        saver = tf.train.Saver()
        print("running")
        # saver = tf.train.import_meta_graph(trained_model_graph)
        saver.restore(sess, trained_model_checkpoint)
        print("Model restored")

        y = sess.run(y_prob, feed_dict={x: image, hold_prob: 1.0})
        return y




def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0,1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    # x --> [batch, H, w, Channels]
    # w --> [filter H, filter W, Channels IN, Channels OUT]

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# CONVOLUTIONAL LAYER
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

# NORMAL (FULLY CONNECTED)
def normal_full_layer(input_layer, size):
    print("input layer shape")
    print(input_layer.get_shape()[1])
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

