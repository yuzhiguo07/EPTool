import os
import numpy as np
import random
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F

# import models.cnn_1d as model
# import models.bilstm as lstmmodel
import models.cbensemble as cbemodel

import loaddata
import loadfeat
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import smile as sm
from smile import flags, logging

flags.DEFINE_string("aln_dpath", "./aln_example",
                    "alignment file directory path")
flags.DEFINE_string("train_fname", "sample.aln",
                    "training alignment file name")
flags.DEFINE_string("valid_fname", "sample.aln",
                    "valid alignment file name")
flags.DEFINE_string("test_fname", "sample.aln",
                    "not required")
flags.DEFINE_integer("feature_size", 42,
                     "sequence feature dim num + pssm feature dim num")
flags.DEFINE_integer("pssm_dim", 21, "pssm feature dim num")
flags.DEFINE_integer("batch_size", 32, "batch size")

flags.DEFINE_string("model_path",
                    "/mnt/new/models/enhance-pssm-checkin/try01", " ")

flags.DEFINE_boolean("load_model", False, "load model from last checkpoint")
flags.DEFINE_boolean("reset_optimizer", True,
                     "reset optimizer, learning rate, and lr scheduler")

flags.DEFINE_integer("cnn_layer_num", 3, "cnn layer number, must >= 3")
flags.DEFINE_integer("cnn_window_size", 3, "cnn window size")
flags.DEFINE_integer("node_size", 100, " ")
flags.DEFINE_float("dropout_rate", 0.0, "dropout rate")
flags.DEFINE_float("output_dropout_rate", 0.0, "cnn output layer dropout")

flags.DEFINE_integer("blstm_layer_num", 2, "blstm layer num")
flags.DEFINE_integer("lstm_hidden_size", 512, "lstm hidden size")
flags.DEFINE_float("lstm_dropout_rate", 0.0, "lstm dropout rate")
flags.DEFINE_float("fc1_dropout_rate", 0.0, "fc layer dropout rate")

flags.DEFINE_float("lr", 0.0001, "learning rate")
flags.DEFINE_string("MSlr_milestones", "30,50,100",
                    "muti step learning rate scheduler")
flags.DEFINE_integer("num_epochs", 400, " ")
flags.DEFINE_float("random_ratio", 0.2, " ")
flags.DEFINE_float("add_ratio", 0.1, " ")

FLAGS = flags.FLAGS


class Config:

    # data
    datapath = FLAGS.aln_dpath
    train_file_list = [FLAGS.train_fname]
    # train_file_list = ['sample.aln'] # debug
    valid_file_list = [FLAGS.valid_fname]
    # valid_file_list = ['sample.aln']
    test_file_list = [FLAGS.test_fname]
    # test_file_list = ['sample.aln']
    data_max_len = 700
    feature_size = FLAGS.feature_size
    batch_size = FLAGS.batch_size

    model_path = FLAGS.model_path
    load_model = FLAGS.load_model
    reset_optimizer = FLAGS.reset_optimizer

    random_ratio = FLAGS.random_ratio
    add_ratio = FLAGS.add_ratio

    # 1d-CNN
    cnn_layer_num = FLAGS.cnn_layer_num  # >= 3
    mid_w_size = FLAGS.cnn_window_size
    bot_window_size = mid_w_size
    window_sizes = [mid_w_size] * int(cnn_layer_num - 2)
    top_window_size = mid_w_size
    node_size = FLAGS.node_size
    dropout_rate = FLAGS.dropout_rate

    # LSTM
    device = 'cuda'
    blstm_layer_num = FLAGS.blstm_layer_num
    lstm_hidden_size = FLAGS.lstm_hidden_size
    lstm_dropout_rate = FLAGS.lstm_dropout_rate
    fc1_dropout_rate = FLAGS.fc1_dropout_rate
    fc1_dim = FLAGS.pssm_dim

    output_dropout_rate = FLAGS.output_dropout_rate

    lr = FLAGS.lr
    weight_decay = 1e-5
    min_lr = 0.00000000001
    num_epochs = FLAGS.num_epochs

    lr_sche = 'MSlr'
    MSlr_milestones = [int(i) for i in FLAGS.MSlr_milestones.split(',')]


def main(_):
    config = Config()
    start = datetime.datetime.now()
    print("Model Structure Parameter: ", Config.__dict__)
    print('================================')

    train_matrix = loaddata.read_aln(
        config, config.train_file_list)
    valid_matrix = loaddata.read_aln(
        config, config.valid_file_list)
    # test_matrix = loaddata.read_aln(
    #     config, config.test_file_list)

    mymodel = cbemodel.CBE(config)
    print(mymodel)

    if torch.cuda.is_available():
        print('cuda is available')
        mymodel = mymodel.cuda()

    optimizer = torch.optim.Adam(
        mymodel.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.lr_sche == 'RlrOP':
        lr_scheduler = ReduceLROnPlateau(
            optimizer, factor=0.2, patience=3, min_lr=config.min_lr)
    elif config.lr_sche == 'MSlr':
        lr_scheduler = MultiStepLR(optimizer, config.MSlr_milestones)

    start_epoch = 0
    if os.path.exists(os.path.join(config.model_path, 'Epoch_last.ckpt')) and config.load_model:
        checkpoint = torch.load(os.path.join(
            config.model_path, 'Epoch_last.ckpt'))
        mymodel.load_state_dict(checkpoint['mymodel'])
        if not config.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    best_valid_rmse = 999999
    best_epoch = 0
    for epoch in range(start_epoch, config.num_epochs):
        mymodel.train()
        x_batches, y_batches, mask_batches = loaddata.get_batch(
            config,
            train_matrix,
            batch_size=config.batch_size,
            shuffle=True,
            debug=False)
        epoch_loss_sum = 0
        epoch_rmse = 0

        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)

        # for step, (batch_x, batch_y, batch_ss) in enumerate(train_loader):
        for step, (batch_x, batch_y, mask) in enumerate(
                zip(x_batches, y_batches, mask_batches)):
            optimizer.zero_grad()
            outputs = mymodel(batch_x.to('cuda'))

            loss_matrix = nn.MSELoss(reduce=False)(
                outputs, batch_y[:, 21:, :].to('cuda'))

            loss_matrix = loss_matrix.mul(mask[:, 21:, :].to('cuda'))
            loss = torch.sum(loss_matrix) / torch.nonzero(
                mask[:, 21:, :]).size(0)
            epoch_loss_sum += loss
            rmse = torch.sum(torch.sqrt(loss_matrix,
                                        out=None)) / torch.nonzero(
                                            mask[:, 21:, :]).size(0)
            epoch_rmse += rmse
            loss.backward()
            optimizer.step()
            # print('Step [{}/{}], Loss: {}, RMSE:{}'.format(step + 1, epoch+1, loss, rmse))

        lr_scheduler.step()

        state = {'mymodel': mymodel.state_dict(), 'optimizer': optimizer.state_dict(
        ), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch}

        # valid RMSE
        valid_rmse_sum = 0
        valid_num = 0
        valid_x_batches, valid_y_batches, valid_mask_batches = loaddata.get_batch(
            config, valid_matrix, batch_size=1, shuffle=False, debug=False)

        with torch.no_grad():
            mymodel.eval()
            for valid_x, valid_y, valid_mask in zip(
                    valid_x_batches, valid_y_batches, valid_mask_batches):
                outputs_val = mymodel(valid_x.to('cuda'))

                loss_matrix_val = nn.MSELoss(reduce=False)(
                    outputs_val, valid_y[:, 21:, :].to('cuda')).mul(
                        valid_mask[:, 21:, :].to('cuda'))
                valid_rmse_sum += torch.sum(torch.sqrt(
                    loss_matrix_val, out=None)) / torch.nonzero(
                        valid_mask[:, 21:, :]).size(0)
                valid_num += 1

            # test RMSE
            # rmse_test = 0
            # t_test = 0
            # test_x_batches, test_y_batches, test_mask_batches = loaddata.get_batch(
            #     config, test_matrix, batch_size=1, shuffle=False, debug=False)
            # for test_x, test_y, test_mask in zip(
            #         test_x_batches, test_y_batches, test_mask_batches):
            #     outputs_val = mymodel(test_x.to('cuda'))
            #     loss_matrix_val = nn.MSELoss(reduce=False)(
            #         outputs_val, test_y[:, 21:, :].to('cuda')).mul(
            #             test_mask[:, 21:, :].to('cuda'))
            #     rmse_test += torch.sum(torch.sqrt(
            #         loss_matrix_val, out=None)) / torch.nonzero(
            #             test_mask[:, 21:, :]).size(0)
            #     t_test += 1

        valid_rmse = valid_rmse_sum / valid_num
        if valid_rmse < best_valid_rmse:
            best_epoch = epoch + 1
            best_valid_rmse = valid_rmse

        torch.save(
            state,
            os.path.join(config.model_path, 'Epoch_{}.ckpt'.format(
                epoch + 1)))
        torch.save(
            state,
            os.path.join(config.model_path, 'Epoch_last.ckpt'))

        epoch_loss = epoch_loss_sum / (step + 1)
        now = datetime.datetime.now()
        print('Epoch [{}/{}], Best_epoch: {}, Loss: {:5f}, RMSE:{:5f}, Valid_RMSE: {:5f}, time: {}'.format(
            epoch + 1, config.num_epochs, best_epoch, epoch_loss,
            epoch_rmse / (step + 1), valid_rmse, now - start))

        # print('Valid_RMSE: {}, Test_RMSE: {}'.format(valid_rmse_sum / valid_num,
        #                                              rmse_test / t_test))


if __name__ == "__main__":
    # main()
    sm.app.run()
