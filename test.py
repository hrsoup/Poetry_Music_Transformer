import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim
from tensorflow.python import pywrap_tensorflow
import math

from config import *
from GPT_Model import *
from data_pipeline import *

def defineArgs():
    """define args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", choices = ['random_poetry', 'poetry', 
                                                    'poetry_pos', 'poetry_tone',
                                                    'random_music', 'music'])
    return parser.parse_args()

if __name__ == "__main__":

    args = defineArgs()

    testData = MUSIC('../Music_dataset_after_preprocess/test')
    data_test_files = testData.musicVector
    EventDim = IntervalDim + NoteOnDim + NoteOffDim # 356

    hparams = parameters(EventDim, EmbeddingDim, Heads, Layers, Time)

    load_dir = '../Exp_' + str(exp_number) + '/' + str(args.type)+ '_fintune'

#------------------------------------------Draw main graph-------------------------------------#
    tf.reset_default_graph()

    tf.compat.v1.disable_eager_execution()
    X = tf.placeholder(tf.int32, [None, hparams.n_time])
    Y = tf.placeholder(tf.int32, [None, hparams.n_time])

    logits = model(hparams, X)['logits']
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    global_step = tf.Variable(0, name='global_step')
    learning_rate = tf.Variable(1e-4, name='learning_rate')

    # GPU number to use
    gpu_options = tf.GPUOptions(visible_device_list="0")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.global_variables_initializer())

    print('graph create')
    print(args.type)

#--------------------------Load model--------------------------#

    # Restore all paremeters of transformer
    saver = tf.train.Saver()

    restore_file = tf.train.latest_checkpoint(load_dir)
    print(restore_file)
    if restore_file is not None:
        saver.restore(sess, restore_file)
        print("Model restored.", restore_file)
    else:
        print('model not exist.')

#----------------------------------------Test------------------------------------------------#

    inputs = []
    targets = []

    l = len(data_test_files)

    for i in range(l): 
        while(True):
            x_test, y_test = get_data(hparams.n_time, data_test_files, 'test', i, 'music', EventDim)
            if(x_test.shape == y_test.shape):
                break       
        inputs.append(x_test)
        targets.append(y_test)
        
    inputs = np.stack(inputs)
    targets = np.stack(targets)

    test_loss = sess.run(loss, feed_dict={X: inputs, Y: targets})
    test_perplexity = math.exp(test_loss)
    print('test_perplexity is {}'.format(test_perplexity))