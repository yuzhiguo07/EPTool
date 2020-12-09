import random
import torch


def read_data(data_path, label_data, config):
    feature_list = []
    ss_list = []
    with open(data_path) as f:
        feature_tmp = []
        ss_tmp = []
        for index, line in enumerate(f):
            ll = line.split(' ')
            if len(ll) > 1:
                ll = list(map(float, ll[:-1]))
                if len(ll) != config.feature_size:
                    raise selfException(
                        "the length of ll is not {}".format(config.feature_size))
                feature_tmp.append(ll)
            elif(int(float(ll[0].strip())) < 9):
                ss_tmp.append(int(float(ll[0].strip())))

            # if len(feature_tmp) == config.data_max_len:
            if len(ss_tmp) == config.data_max_len:
                feature_list.append(feature_tmp)
                feature_tmp = []
                ss_list.append(ss_tmp)
                ss_tmp = []

    return torch.tensor(feature_list), torch.tensor(ss_list)
