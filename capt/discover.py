import sys
import numpy as np
import tensorflow as tf

from PIL import Image
from PIL import ImageFile
from cnn_sys import crack_captcha_cnn
from predict import hack_function
from cfg import MAX_CAPTCHA, CHAR_SET_LEN, model_path
from utils import convert2gray, vec2text

ImageFile.LOAD_TRUNCATED_IMAGES = True

def discern( imgFile ):
    output = crack_captcha_cnn()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        image = Image.open( imgFile )
	image = image.resize((160,60))
        image = np.array(image)
        image = convert2gray(image)
        image = image.flatten() / 255

        captcha = hack_function(sess, predict, image)
        return captcha

if __name__ == '__main__':
    imgFile = sys.argv[1]
    captcha = discern( imgFile )
    print( captcha )
