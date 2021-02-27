import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim
from tensorflow.python import pywrap_tensorflow
import math

from config import *
from GPT_Model import *
from data_pipeline import *

mode = "finetune"
type = 'music'

def defineArgs():
    """define args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--pretrain_data", choices = ['random_poetry', 'poetry', 
                                                    'poetry_pos', 'poetry_tone',
                                                    'random_music', 'music'])
    return parser.parse_args()

if __name__ == "__main__":

    args = defineArgs()

    trainData = MUSIC('../Music_dataset_after_preprocess/train')
    data_train_files = trainData.musicVector
    EventDim = IntervalDim + NoteOnDim + NoteOffDim # 356

    hparams = parameters(EventDim, EmbeddingDim, Heads, Layers, Time)

    load_dir = '../Exp_' + str(exp_number) + '/' + str(args.pretrain_data)+ '_pretrain' 
    save_dir = '../Exp_' + str(exp_number) + '/' + str(args.pretrain_data)+ '_fintune'
    
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

    optimizer = tf.train.AdamOptimizer(learning_rate)
    output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='linear1|linear2')
    train_step = optimizer.minimize(loss, var_list=output_vars, global_step=global_step)

    # GPU number to use
    gpu_options = tf.GPUOptions(visible_device_list="0")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.global_variables_initializer())

    print('graph create')
    print(type)
    print(mode)

#--------------------------Load model if exist && TensorboardX Logger--------------------------#

    # only restore paremeters of transformer
    sess.run(tf.global_variables_initializer())
    ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='transformer')
    saver = tf.train.Saver(ref_vars)

    restore_file = tf.train.latest_checkpoint(load_dir)
    print(restore_file)
    if restore_file is not None:
        saver.restore(sess, restore_file)
        print("Model restored.", restore_file)
    else:
        print('model not exist.')

    # redefine saver, save all parameters of transformer
    saver_all = tf.train.Saver()

    # Logger
    from tensorboardX import SummaryWriter

    class Logger(SummaryWriter):
        def __init__(self, logdir):
            super(Logger, self).__init__(logdir)

        def log(self, log_string, value, iteration):
                self.add_scalar(log_string, value, iteration)
                
    logger = Logger(save_dir)  

#----------------------------------------Train------------------------------------------------#
    print('iteration\t', 'loss\t', 'train_perplexity\t')
    while(True):
        for _ in range(100):
            _inputs = []
            _targets = []
            for _ in range(batch_size):
                while(True):
                    x, y = get_data(hparams.n_time, data_train_files,'train', 0, type, EventDim)
                    if(x.shape == y.shape):
                        break
                    
                _inputs.append(x)
                _targets.append(y)
            _inputs = np.stack(_inputs)
            _targets = np.stack(_targets)
            
            _, _global_step, _loss = sess.run([train_step, global_step, loss], 
                                            feed_dict={X: _inputs, 
                                                        Y: _targets,
                                                        learning_rate: 1e-4})
            
            train_perplexity = math.exp(_loss) # log perplexity is equal to perplexity 
            
            if _global_step % 10 == 0:
                logger.log('loss', _loss, _global_step)
                print(str(_global_step)+'\t', str(_loss)+'\t', str(train_perplexity)+'\t')
            
            if _global_step % 100 == 0:
                save_path = saver_all.save(sess, save_dir + '/checkpoint', global_step=_global_step)
                print("Model saved in path: %s" % save_path)