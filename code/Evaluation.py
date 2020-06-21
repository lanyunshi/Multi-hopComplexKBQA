import re
import os
import csv
import numpy as np
import random
from tool import *
import json
import argparse
from collections import defaultdict

#random.seed(123)

def evaluation(data_path, file):
    a = []
    folder = os.path.basename(data_path)
    with open('data/test_%s/a.txt'%folder) as f:
        for line_idx, line in enumerate(f):
            a += [line.strip().lower().split('\t')]

    accuracies, precisions, recalls, F1s, hit1s = [], [], [], [], []
    with open(os.path.join(data_path, '%s_predcp.txt' %file)) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if len(line.split('\t')) > 3:
                idx, g, F = line.split('\t')[:3]
                F = float(F[1: -1])
                ans = set(line.split('\t')[3:])
            else:
                ans = set([])
            acc, precision, recall, F1, hit1 = generate_evaluation_tmp(ans, set(a[line_idx]))
            #print(ans, a[line_idx], F1)
            #if line_idx == 77: print(F1); exit()
            accuracies += [acc]
            precisions += [precision]
            recalls += [recall]
            F1s += [F1]
            hit1s += [hit1]

    return np.mean(hit1s), np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(F1s)

def loop_evaluation(data_path, file):
    max_hit1, max_accuracy, max_precision, max_recall, max_F1 = 0, 0, 0, 0, 0
    for _ in range(1):
        hit1s, accuracies, precisions, recalls, F1s = evaluation(data_path, file)
        if hit1s > max_hit1:
            max_hit1, max_accuracy, max_precision, max_recall, max_F1 = hit1s, accuracies, precisions, recalls, F1s
    print('Hit@1: %s\nAccuracy: %s\nPrecision: %s\nRecall: %s\nF1s: %s' %(max_hit1, max_accuracy, max_precision, max_recall, max_F1))


def convert_ans_to_anstxt_CWQ(data_path, ans):
    q2id = json.load(open('data/CWQ/q2id.json', 'r'))
    outputs, M2N = [], json.load(open('data/WBQ/m2n.json', "r"))
    with open('data/test_CWQ/q.txt') as f:
        for line_idx, line1 in enumerate(f):
            line1 = line1.strip().lower()
            line2 = ans[line_idx]
            id = q2id[line1]
            if line2 not in M2N: update_m2n(line2, M2N)
            output = {"ID": id, "answer": M2N[line2].lower()}
            outputs += [output]
    g = open(os.path.join(data_path, 'CWQ_final_submit.json'), 'w')
    json.dump(outputs, g)
    g.close()


def generate_hits1_result_from_json(data_path, file):
    a = []
    folder = os.path.basename(data_path)
    with open('data/test_%s/a.txt'%folder) as f:
        for line_idx, line in enumerate(f):
            a += [line.strip().lower().split('\t')]

    hits1, total_ans = [], []
    with open(os.path.join(data_path, '%s_predcp.json' %file)) as f:
        for line_idx, line in enumerate(f):
            line = json.loads(line)
            ans = defaultdict(int)

            new_line = {}
            for p in line:
                for an in line[p]:
                    new_line[an] = line[p][an]
            line = new_line

            one_ans = ''
            if len(line) == 0:
                hits1 += [0]
            else:
                last_idx = np.max([int(an[0]) for an in line])
                for an in line:
                    an_tmp = re.sub('^[012]\d', '', an)
                    if re.search('^[012]\d', an):
                        ans[an_tmp] += line[an]
                #print(ans)
                score = sorted(ans.values())[::-1][0]
                ans = set([an for an in ans if ans[an] == score])

                one_ans = random.sample(ans, 1)[0]
                hit1 = int(one_ans in a[line_idx])
                hits1 += [hit1]

            total_ans += [one_ans]
    #print(hits1)
    print('Hit@1: %s' %np.mean(hits1))
    return total_ans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_path", default=None, type=str, help="Path to data")
    parser.add_argument("--data_file", default=None, type=str, help="File of pred_cp data")
    parser.add_argument("--mode", default='eval', type=str, help="eval: evaluate [hit@1|accuracy|precision|recall|f1]; trans: transform the pred_cp file to official evaluation file")
    args = parser.parse_args()

    data_path = args.data_path
    data_file = args.data_file
    print(data_path, data_file)

    if args.mode == 'eval':
        loop_evaluation(data_path, data_file)
    elif args.mode == 'trans':
        total_ans = generate_hits1_result_from_json(data_path, data_file)
        convert_ans_to_anstxt_CWQ(data_path, total_ans)


'''
python code/Evaluation.py \
    --data_path trained_model/CQ \
    --data_file new_Best \
    --mode eval \

python code/Evaluation.py \
    --data_path trained_model/CWQ \
    --data_file Bert_Best_2_2_100 \
    --mode trans \
'''
