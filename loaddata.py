import torch
import random
import os
import time
import numpy as np
import copy

import smile as sm
from smile import flags, logging

seq_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
            'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', '*', '-']
seq_dict = {}

for i, seq_k in enumerate(seq_list):
    seq_dict[seq_k] = i


def read_aln(config, aln_file_list):

    aln_folder = config.datapath

    def is_float(s):
        try:
            a = float(s)
        except:
            return False
        return True

    matrix_dict = {}

    for aln_file in aln_file_list:
        aln_path = os.path.join(aln_folder, aln_file)
        with open(aln_path) as f:
            i = 0
            lines = f.readlines()
            for index, line in enumerate(lines):
                if index != len(lines) - 1 and is_float(lines[index+1]):
                    i += 1
                    seq = line.strip()
                    raw_num = int(float(lines[index+1]))
                    if raw_num < 1:
                        print(
                            "Protein_seq: {} didn't match any protein in Database".format(seq))
                        continue
                    raws = lines[index+2: index+2+raw_num]
                    matrix = []  # the i'th line is the msa_feature list for the i'th char in the seq
                    for idx in range(len(seq)):
                        in_matrix = []
                        for raw in raws:
                            try:
                                in_matrix.append(seq_dict[raw[idx]])
                            except:
                                in_matrix.append(seq_dict['X'])
                        matrix.append(in_matrix)
                    matrix_dict[seq] = matrix
        print(aln_file+' Done.')
    return matrix_dict


def get_batch(config, matrix_dict, batch_size, shuffle=False, debug=False, eval=False):
    key_list = list(matrix_dict.keys())
    # print(key_list)
    if shuffle:
        random.shuffle(key_list)

    def get_pssm(matrix_dict, random_ratio, add_ratio=0.1):
        # seq_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','*','-']
        basic_pssm_dict = {}
        for index, key in enumerate(matrix_dict):
            # get ont-hot feature for key
            num_class = len(seq_list) - 2
            key_arr = [seq_dict[k] for k in key]
            one_hot_fearture = np.eye(num_class)[key_arr]

            feat_matrix = []
            randnum = random.randint(0, 100)
            matrix = np.array(matrix_dict[key])  # .astype(np.int64)
            m_line_num = len(matrix_dict[key][0])
            if random_ratio != 1:
                # new_num = int(m_line_num * random_ratio)

                random_ratio_new = random.uniform(
                    random_ratio, random_ratio+add_ratio)
                new_num = int(m_line_num * random_ratio_new) + 1

                index_lines = np.array([i for i in range(m_line_num)])

                index_random_list = np.random.choice(
                    index_lines, new_num, False)
                matrix = np.transpose(np.transpose(
                    matrix)[index_random_list, :])
                m_line_num = new_num

            frequency_matrix = np.apply_along_axis(lambda x: np.bincount(
                x, minlength=len(seq_list)), axis=1, arr=matrix)
            frequency_matrix = np.split(
                frequency_matrix.astype(np.float64), [-2], axis=1)[0]
            pseufocount = 1.0
            N = m_line_num
            score = (frequency_matrix + pseufocount)/(N + 20 * pseufocount)

            # new
            bf_list = np.array([0.0 for i in range(len(seq_list))])
            for row in matrix:
                bf_list += np.bincount(row, minlength=len(seq_list))

            bf_list = np.split(bf_list.astype(np.float64), [-2])[0]
            b_all = 0
            for b in bf_list:
                b_all += b
            bf_list_avg = bf_list / b_all
            bf_list_avg = bf_list_avg.tolist()
            bf = []
            for _ in range(score.shape[0]):
                bf.append(bf_list_avg)
            bf = np.array(bf)

            # pssm = log(score/0.05)   pssm = log(score/bf)
            score = np.log(score/bf)
            # pssm = Sigmoid(pssm)  s = 1 / (1 + np.exp(-c))
            score = 1 / (1 + np.exp(-score))

            # pssm = log(score/0.05)
            feature_matrix = np.concatenate((one_hot_fearture, score), axis=1)

            basic_pssm_dict[key] = feature_matrix.tolist()

        return basic_pssm_dict

    def padding(data_max_len, matrix, feature_size):
        mask_dict = {}
        mask_list = [0.0 for _ in range(
            int(feature_size/2))] + [1.0 for _ in range(int(feature_size/2))]
        padding_list = [0.0 for _ in range(feature_size)]
        for key in matrix:
            mask_dict[key] = []
            key_len = len(key)
            padding_num = data_max_len - key_len
            for _ in range(key_len):
                mask_dict[key].append(mask_list)
            for _ in range(padding_num):
                matrix[key].append(padding_list)
                mask_dict[key].append(padding_list)
        return matrix, mask_dict

    data_matrix_dict = get_pssm(matrix_dict, config.random_ratio)
    label_matrix_dict = get_pssm(matrix_dict, 1)
    if eval:
        data_matrix_dict = copy.deepcopy(label_matrix_dict)
    else:
        data_matrix_dict = get_pssm(
            matrix_dict, config.random_ratio, config.add_ratio)

    data_max_len = config.data_max_len
    feature_size = config.feature_size

    data_matrix_dict, data_mask_dict = padding(
        data_max_len, data_matrix_dict, feature_size)
    label_matrix_dict, label_mask_dict = padding(
        data_max_len, label_matrix_dict, feature_size)

    x_batches, y_batches, mask_batches = [], [], []
    for _ in range(int(len(key_list)/batch_size) + 1):
        x_batch, y_batch, mask_batch = [], [], []
        for key in key_list[:batch_size]:
            x_batch.append(data_matrix_dict[key])
            y_batch.append(label_matrix_dict[key])
            mask_batch.append(data_mask_dict[key])

        x_batch = torch.tensor(x_batch, dtype=torch.float).permute(0, 2, 1)
        y_batch = torch.tensor(y_batch, dtype=torch.float).permute(0, 2, 1)
        mask_batch = torch.tensor(
            mask_batch, dtype=torch.float).permute(0, 2, 1)

        x_batches.append(x_batch)
        y_batches.append(y_batch)
        mask_batches.append(mask_batch)
        key_list = key_list[batch_size:]
        if len(key_list) == 0:
            break
    return x_batches, y_batches, mask_batches
