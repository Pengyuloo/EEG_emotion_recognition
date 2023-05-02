# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import scipy.io as sio
import time, random, argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from lstm_capsule import lstmCapsule
import torch
import random
import codecs
from sklearn import decomposition
from tensorboardX import SummaryWriter
import datetime
now_time = datetime.datetime.now().strftime('%Y-%m-%d')
import dataManager

parser = argparse.ArgumentParser()
parser.add_argument('--save_file', type=str)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--sub_id', type=int)
parser.add_argument('--clip_id', type=int)
parser.add_argument('--session_id', type=int)
parser.add_argument('--path', type=str)
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--db_name', type=str, choices=["SEED"])
parser.add_argument('--strategy', type=str, choices=["loso"])
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--cell_dropout', type=float, default=0.1,choices=[0.5,0.1])
parser.add_argument('--final_dropout', type=float, default=0.1,choices=[0.5,0.1])
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--iter_num', type=int, default=32*256)
parser.add_argument('--per_checkpoint', type=int, default=32)
parser.add_argument('--seed', type=int, default=1705216)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument('--bidirectional', type=bool, default=False)
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
parser.add_argument('--name_model', type=str, default='EEG-Emotion-master')
FLAGS = parser.parse_args()


torch.backends.cudnn.enabled = False
use_cuda = torch.cuda.is_available()

#np.random.seed(FLAGS.seed)
#random.seed(FLAGS.seed)
#torch.manual_seed(FLAGS.seed)
#if use_cuda:
#    torch.cuda.manual_seed_all(FLAGS.seed)
    
channels, word_dim, hidden_dim, n_label = 200, 310, 512, 3

def train(model, train_data, train_label):
    train_num = train_data.shape[0]
    index = [random.randint(0, train_num-1) for i in range(FLAGS.batch_size)]
    batched_data = train_data[index]
    batched_label = train_label[index]
    loss, _ = model.stepTrain(batched_data, batched_label)
    return loss

def evaluate(model, data, label):
    loss = np.zeros((1, )) 
    st, ed, times = 0, FLAGS.batch_size, 0
    pred_matrix = []
    y_true = []
    length= data.shape[0]
    while st < length:
        batched_data = data[st:ed]
        batched_label = label[st:ed]
        outputs, pred_ = model.stepTrain(batched_data, batched_label, inference=True)
        pred_matrix.extend(pred_)
        y_true.extend(batched_label)
        loss += outputs
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    pred_vector = np.argmax(np.array(pred_matrix), axis=1)
    c_m = confusion_matrix(np.array(y_true), pred_vector, labels=range(n_label))
    loss /= times
    accuracy = np.sum([c_m[i][i] for i in range(n_label)]) / np.sum(c_m)
    
    return loss, accuracy, c_m


if __name__ == "__main__":
    ##load data and split traning & testing
    if FLAGS.strategy == "loso":
        data = sio.loadmat(FLAGS.path)
        feature_arr = data['feature_arr'] 
        label_arr = data['label_arr']
        label_arr = np.reshape(label_arr, [-1,])
        feature_arr = np.reshape(feature_arr, [-1, ])
        # print("feature_arr.shape:",feature_arr.shape)


    if FLAGS.strategy=="loso":
        train_data, test_data, train_label, test_label = dataManager.leave_one_sub_out(feature_arr, label_arr, FLAGS.sub_id, FLAGS.db_name)

    
    print('model parameters: %s' % str(FLAGS))
    print("Use cuda: %s" % use_cuda)
    print('train data: %s, test data: %s' % (train_data.shape, test_data.shape))
    
    model = lstmCapsule(
            dim_input=word_dim,
            dim_hidden=hidden_dim,
            channels=channels,
            n_layers=FLAGS.n_layer,
            n_label=n_label,
            batch_size = FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            weight_decay=FLAGS.weight_decay,
            cell_dropout_rate=FLAGS.cell_dropout,
            final_dropout_rate=FLAGS.final_dropout,
            optim_type=FLAGS.optim_type,
            rnn_type=FLAGS.rnn_type,
            bidirectional=FLAGS.bidirectional,
            use_cuda=use_cuda)

    writer = SummaryWriter(log_dir= FLAGS.log_dir)
    model.init()

    
    loss_step, time_step = np.zeros((1,)), 1e10
    start_time = time.time()
    train_max_acc = 0
    test_max_acc = 0
    train_max_step = 0
    test_max_step = 0

    result = codecs.open(FLAGS.save_file, 'a', 'utf-8')
    
    if FLAGS.strategy=="loso":
        result.write('========\nsubject %s\n========\n' % str(FLAGS.sub_id))

    result.write('model parameters: %s' % str(FLAGS))
    result.close()
    for step in range(FLAGS.iter_num):
        # p = float( step * 1) / FLAGS.iter_num
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        if step % FLAGS.per_checkpoint == 0:
            show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
            time_step = time.time() - start_time
            print("------------------------------------------------------------------")
            print('Time of iter training %.2f s' % time_step)
            print("On iter step %s:, global step %d learning rate %.4f Loss-step %s"
                    % (step/FLAGS.per_checkpoint, step, model.optimizer.param_groups[0]['lr'], show(np.exp(loss_step))))

            loss1, acc1, c_m1 = evaluate(model, train_data, train_label)
            print('In dataset train: Loss is %s, Accuracy is %s' % (show(np.exp(loss1)), acc1))
            print('%s' % c_m1)

            writer.add_scalar('train/loss', loss1[0], step)
            writer.add_scalar('train/acc', acc1, step)

            if acc1 >= train_max_acc:
                train_max_acc = acc1
                train_max_step = step

            loss2, acc2, c_m2 = evaluate(model, test_data, test_label)
            
            if acc2 >= test_max_acc:
                test_max_acc = acc2
                test_max_step = step
                
            print('In dataset test: Loss is %s, Accuracy is %s' % (show(np.exp(loss2)), acc2))
            print('Confusion matrix: ')
            print('%s' % c_m2)
            writer.add_scalar('test/loss', loss2[0], step)
            writer.add_scalar('test/acc', acc2, step)

            if FLAGS.strategy=="loso":
                writer.add_scalars('loss_group'+str(FLAGS.sub_id), {'train/loss': loss1[0],
                                                    'test/loss': loss2[0]}, step)
                writer.add_scalars('acc_group'+str(FLAGS.sub_id), {'train/acc': acc1,
                                                 'test/acc': acc2}, step)


            start_time = time.time()
            loss_step = np.zeros((3, ))

            print('Maximum train accuracy is %s in global step %s' % (train_max_acc, train_max_step))
            print('Maximum test accuracy is %s in global step %s' % (test_max_acc, test_max_step))

            ##### write in file #####
            result = codecs.open(FLAGS.save_file, 'a', 'utf-8')
            result.write("\n------------------------------------------------------------------\n")
            result.write('Time of iter training %.2f s\n' % time_step)
            result.write("On iter step %s:, global step %d learning rate %.4f Loss-step %s\n"
                  % (step / FLAGS.per_checkpoint, step, model.optimizer.param_groups[0]['lr'], show(np.exp(loss_step))))
            result.write('In dataset train: Loss is %s, Accuracy is %s\n' % (show(np.exp(loss1)), acc1))
            result.write('Confusion matrix: \n')
            result.write('%s\n' % c_m1)
            result.write('In dataset test: Loss is %s, Accuracy is %s\n' % (show(np.exp(loss2)), acc2))
            result.write('Confusion matrix: \n')
            result.write('%s\n' % c_m2)
            result.write('Maximum train accuracy is %s in global step %s\n' % (train_max_acc, train_max_step))
            result.write('Maximum test accuracy is %s in global step %s\n' % (test_max_acc, test_max_step))
            result.close()

        loss_step += train(model, train_data, train_label)/FLAGS.per_checkpoint
#     attention_result = codecs.open(FLAGS.save_file+'attention', 'a', 'utf-8')
#     attention_result.write('===========\nsubject %s\n===========\ntrain max \n%s\ntest max\n%s\n\n'% (i, train_max_attention, test_max_attention))
#     attention_result.close() 
    writer.export_scalars_to_json(FLAGS.log_dir+"/test.json")
    writer.close()



