import os
import numpy as np
import random

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
# import loaddata.get_batch as get_batch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import smile as sm
from smile import flags, logging


flags.DEFINE_string("eval_feat_path", "./feat_example/sample2.feat",
                    "To generate the original feat file, the PSSM calculatation must follow the PSSM fomulas in our paper")
flags.DEFINE_string("save_fpath",
                    "/mnt/new/models/enhance-pssm-checkin/try01/save_new_feat/new.feat", "")
flags.DEFINE_string("model_path",
                    "/mnt/new/models/enhance-pssm-checkin/try01", " ")

flags.DEFINE_integer("epoch", 2, "eval checkpoint number")

flags.DEFINE_integer("feature_size", 42,
                     "sequence feature dim num + pssm feature dim num")
flags.DEFINE_integer("pssm_dim", 21, "pssm feature dim num")
flags.DEFINE_integer("batch_size", 32, "batch size")

flags.DEFINE_boolean("load_model", False, " ")
flags.DEFINE_boolean("reset_optimizer", True, " ")

flags.DEFINE_integer("cnn_layer_num", 3, ">= 3")
flags.DEFINE_integer("cnn_window_size", 3, " ")
flags.DEFINE_integer("node_size", 100, " ")
flags.DEFINE_float("dropout_rate", 0.0, " ")
flags.DEFINE_float("output_dropout_rate", 0.0, " ")

flags.DEFINE_integer("blstm_layer_num", 2, " ")
flags.DEFINE_integer("lstm_hidden_size", 512, " ")
flags.DEFINE_float("lstm_dropout_rate", 0.0, " ")
flags.DEFINE_float("fc1_dropout_rate", 0.0, " ")

flags.DEFINE_float("lr", 0.0001, " ")
flags.DEFINE_string("MSlr_milestones", "30,50,100", " ")
flags.DEFINE_integer("num_epochs", 400, " ")
flags.DEFINE_float("random_ratio", 0.2, " ")
flags.DEFINE_float("add_ratio", 0.1, " ")

FLAGS = flags.FLAGS


class Config:

    # data
    data_max_len = 700
    feature_size = FLAGS.feature_size
    batch_size = FLAGS.batch_size

    model_path = FLAGS.model_path

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


def main(_):
    epoch = FLAGS.epoch
    save_fpath = FLAGS.save_fpath

    config = Config()

    mymodel = cbemodel.CBE(config)
    print(mymodel)

    if torch.cuda.is_available():
        print('cuda is available')
        mymodel = mymodel.cuda()

    checkpoint = torch.load(os.path.join(
        config.model_path, 'Epoch_{}.ckpt'.format(epoch)))
    mymodel.load_state_dict(checkpoint['mymodel'])
    mymodel.eval()
    eval_data, eval_ss = loadfeat.read_data(
        FLAGS.eval_feat_path, FLAGS.eval_feat_path, config)
    eval_data = eval_data.permute(0, 2, 1)

    # put dataset in DataLoader

    eval_torch_dataset = Data.TensorDataset(eval_data, eval_ss)
    eval_loader = Data.DataLoader(
        dataset=eval_torch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    with open(save_fpath, 'w') as w:
        for index, (batch_x, batch_ss) in enumerate(eval_loader):
            outputs = mymodel(batch_x.to('cuda'))
            # (h0, c0) = hidden_states_0
            mask = (batch_ss != 8).long().float()
            mask = mask.resize(outputs.shape[0], config.data_max_len, 1)
            mask = mask.expand(outputs.shape[0], config.data_max_len, int(
                config.feature_size/2)).permute(0, 2, 1)
            outputs = outputs.mul(mask.to('cuda'))

            x_onehot = batch_x.to('cuda')[:, :21, :]

            new_feat = torch.cat((x_onehot, outputs), 1).permute(
                0, 2, 1)[0].cpu().detach().numpy().tolist()
            ss_feat = batch_ss[0].cpu().numpy().tolist()
            w.write('{}\n'.format(config.data_max_len))
            for line in new_feat:
                for item in line:
                    w.write('{} '.format(item))
                w.write('\n')
            for ss in ss_feat:
                w.write('{}\n'.format(ss))
            print('{}/{}'.format(index+1, len(eval_loader)), end='\r')

    print('\nDone')


if __name__ == "__main__":
    sm.app.run()

