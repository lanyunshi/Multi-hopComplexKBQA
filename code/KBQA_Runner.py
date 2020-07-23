import logging
import numpy as np
from time import gmtime, strftime
import time as mytime
from datetime import datetime
import os
import pickle
import re
import copy
import argparse
import torch
import random
import json
import copy
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm, trange
from torch.distributions import Categorical
import torch.nn.functional as F
from SPARQL_test import *
from tool import *
import sqlite3

from tokenization import Tokenizer
from ModelsRL import ModelConfig, Policy
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
const_minimax_dic = 'first|last|predominant|biggest|major|warmest|tallest|current|largest|most|newly|son|daughter' #

class TrainingInstance(object):
    """A single training instance (A question associated with its topic entities)"""
    def __init__(self, q_idx, raw_question, question, topic_entity, constraint, answer, golden_graph):
        self.q_idx = q_idx
        self.raw_question = raw_question
        self.question = question
        self.topic_entity = topic_entity
        self.answer = answer
        self.const_type = constraint
        self.const_time = re.findall('\d+', raw_question) if re.search('\d+', raw_question) and len(re.findall('\d+', raw_question)[0])==4 else None
        self.const_minimax = re.findall('(?<= )(%s)' %const_minimax_dic, raw_question) if re.search('(?<= )(%s)' %const_minimax_dic, raw_question) else None
        self.golden_graph = golden_graph
        self.current_topic_entity = {}
        self.hop_number = 0
        self.candidate_paths = []
        self.candidate_answers = []
        self.candidate_paths2previous_index = []
        self.candidate_path_index = []
        self.history_candidate_paths = set()
        self.previous_index = {}
        self.current_F1s = []
        self.orig_F1s = []
        self.F1s = []
        self.previous_action_num = 0

        for t in topic_entity:
            self.current_topic_entity[t] = ((t, ), )

    def __str__(self):
        s = ""
        s += "question: %s\n" %self.question
        s += "topic entity: %s\n" %self.topic_entity
        s += "current topic entity: %s\n" %self.current_topic_entity
        s += "hop number: %s\n" %self.hop_number
        s += "answer: %s\n" %' '.join(list(self.answer))
        s += "golden graph: %s\n" %str(self.golden_graph)
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

    def reset(self):
        self.hop_number = 0
        self.current_topic_entity = {}

        for t in self.topic_entity:
            self.current_topic_entity[t] = ((t, ), )

def create_instances(input_file, tokenizer):
    raw_questions, questions, topic_entities, answers, golden_graphs, constraints = [], [], [], [], [], []
    line_idx = 0
    Q_file = os.path.join(input_file, 'q.txt')
    with open(Q_file, 'r') as f: # Load question file
        while True:
            line = f.readline()
            if (not line): # or (line_idx == 10000): # 3000
                break
            line = line.strip()
            raw_questions.append(line)
            tokens = tokenizer.tokenize(line)
            questions.append(tokenizer.convert_tokens_to_ids(tokens)[:20])
            line_idx += 1

    TE_file = os.path.join(input_file, 'te.json')
    with open(TE_file, 'r') as f: # Load detected topic entity file
        for line_idx, line in enumerate(f):
            line = json.loads(line)
            topic_entities.append(line)

    CON_file = os.path.join(input_file, 'con.txt')
    with open(CON_file, 'r') as f: # Load detected constraints file
        for line_idx, line in enumerate(f):
            line = line.strip()
            constraints.append(line.split('\t'))

    A_file = os.path.join(input_file, 'a.txt')
    with open(A_file, 'r') as f: # Load answer file
        for line_idx, line in enumerate(f):
            line = line.strip().lower()
            answers.append(line.split('\t'))

    if input_file.split('_')[-1] not in ['CQ']:
        G_file = os.path.join(input_file, 'g.txt')
        with open(G_file, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                g = re.split('\t| ', line)
                if input_file.split('_')[-1] in ['WBQ', 'CWQ']: g = [w for w in g if not re.search('^\?', w)]
                golden_graphs.append(tuple(g))

    instances = []
    for q_idx, q in enumerate(questions):
        instances.append(TrainingInstance(q_idx=(q_idx + 1),
                                        raw_question=raw_questions[q_idx],
                                        question=questions[q_idx],
                                        topic_entity=topic_entities[q_idx],
                                        constraint=constraints[q_idx],
                                        answer=answers[q_idx],
                                        golden_graph=golden_graphs[q_idx] if golden_graphs else ()))

    return instances

def Load_KB_Files(KB_file):
    """Load knowledge base related files.
    KB_file: {ent: {rel: {ent}}}; M2N_file: {mid: name}; QUERY_file: set(queries)
    """
    KB = json.load(open(KB_file, "r"))
    return KB

def Save_KB_Files(KB, KB_file):
    """Save knowledge base related files."""
    g = open(KB_file, "w")
    json.dump(KB, g)
    g.close()

def relaxed_softmax(logits, temp = 10.):
    logits = logits / temp
    return logits

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def update_raw_candidate_paths(path, ans, prev_idx, candic, batch, time):
    if path not in candic:
        if time > 0: batch.candidate_paths2previous_index += [batch.previous_index[prev_idx]]
        candic[path] = set(ans)
    else:
        candic[path].update(set(ans))

def clean_answer(raw_answer):
    a = list(raw_answer)[0]
    if re.search('^[mg]\.', a):
        return raw_answer
    # elif re.search('T', a):
    #     return set([datetime.strptime(a, "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d") for a in raw_answer])
    else:
        return set([a.lower() for a in raw_answer])

def check_answer(raw_answer):
    a = list(raw_answer)[0]
    if re.search('^[mg]\.', a):
        return 1
    elif a.isdigit() and len(a) == 4:
        return 2
    else:
        try:
            datetime.strptime(a, "%Y-%m-%d")
            return 2
        except:
            if len(raw_answer) < 5 and len(a) < 30:
                return 1
            else:
                return 0

def retrieve_KB(batch, KB, QUERY, M2N, tokenizer, method, train_limit_number=1000, time = 0, is_train = False, save_model='SO'):
    '''Retrieve subgraphs from the KB based on current topic entities'''
    te, c_te, hn = batch.topic_entity, batch.current_topic_entity, batch.hop_number
    raw_candidate_paths, paths, batch.orig_F1s ={}, {}, []
    hn_mark, query_num = 0, 0
    if time > 0:
        for h_idx, h in enumerate(c_te):
            update_raw_candidate_paths(c_te[h], [h], c_te[h], raw_candidate_paths, batch, time)
        batch.previous_action_num = len(raw_candidate_paths)
    for previous_path in set(c_te.values()):
        raw_paths, queries = {}, set()
        if previous_path in KB:
            paths = KB[previous_path]
        if time:
            if tokenizer.dataset in ['CWQ'] and time < 2: #(tokenizer.dataset in ['CWQ'] and time < 2) or (len(paths) == 0 and time==0): #(tokenizer.dataset in ['CWQ'] and time < 2) or
                ''' Single hop relations, remove this when WBQ '''
                path, query = SQL_1hop(previous_path, QUERY=QUERY)
                if query:  raw_paths.update(path); QUERY.update(query); query_num += 1 #QUERY_save.update(query) #
                ''' 2 hop relations, remove this when WBQ'''
                path, query = SQL_2hop(previous_path, QUERY=QUERY)
                if query:  raw_paths.update(path); QUERY.update(query); query_num += 1 #QUERY_save.update(query)
        if time:
            ''' Constraint relations via entities, time, min max'''
            overlap_te = set([mid for mid in te if te[mid][1] == te[previous_path[0][0]][1]]) if tokenizer.dataset in ['CQ'] else set([mid for mid in sum(previous_path, ()) if re.search('^[mg]\.', mid)]) #
            const = set(te.keys()) - overlap_te
            #print(te); print(const); print(batch.const_time); print(batch.const_minimax); print(batch.const_type)
            if len(const): path, query = SQL_1hop_reverse(previous_path, const, QUERY=QUERY) #
            if len(const) and query: raw_paths.update(path); QUERY.update(query); query_num += 1 #QUERY_save.update(query)
            # if len(const): path, query = SQL_2hop_reverse(previous_path, const, QUERY=QUERY)
            # if len(const) and query: raw_paths.update(path); QUERY.update(query); query_num += 1
            if batch.const_time: path, query = SQL_1hop_reverse(previous_path, batch.const_time, QUERY=QUERY)
            if batch.const_time and query: raw_paths.update(path); QUERY.update(query); query_num += 1 #QUERY_save.update(query)
            if batch.const_minimax: path, query = SQL_1hop_reverse(previous_path, batch.const_minimax, QUERY=QUERY)
            if batch.const_minimax and query: raw_paths.update(path); QUERY.update(query); query_num += 1 #QUERY_save.update(query)#
            ''' Constraint relations via types'''
            const = set(batch.const_type)
            if len(const): path, query = SQL_1hop_type(previous_path, const, QUERY=QUERY)
            if len(const) and query:  raw_paths.update(path); QUERY.update(query); query_num += 1 #QUERY_save.update(query) #
            if time:
                for raw_path in copy.deepcopy(raw_paths): raw_paths[previous_path + raw_path] = raw_paths.pop(raw_path)
        for raw_path in raw_paths: update_hierarchical_dic_with_set(KB, previous_path, raw_path, raw_paths[raw_path])
        if len(raw_paths) == 0 and (previous_path not in KB): KB[previous_path] = raw_paths
        paths.update(raw_paths)
        for raw_path in paths:
            path = raw_path
            update_raw_candidate_paths(path, paths[raw_path], previous_path, raw_candidate_paths, batch, time)

    '''Process the candidate paths'''
    candidate_paths, topic_scores, topic_numbers, answer_numbers, type_numbers, superlative_numbers, year_numbers, hop_numbers, F1s, RAs = [], [], [], [], [], [], [], [], [], []
    max_cp_length, types = 0, []
    limit_number = train_limit_number
    for p_idx, p in enumerate(raw_candidate_paths):
        CWQ_F1 = generate_F1_tmp(clean_answer(raw_candidate_paths[p]), batch.answer) if tokenizer.dataset in ['CWQ', 'WBQ'] else 0.
        if (not is_train) or (np.random.random() < (limit_number*1./len(raw_candidate_paths))) or CWQ_F1 > 0.5:
            '''If answer equals to topic entities'''
            if raw_candidate_paths[p] == set([p[0][0]]) or not check_answer(raw_candidate_paths[p]): continue #
            path, topic_score, p_tmp, topic_number, type_number, superlative_number, year_number, contain_minimax, skip = [], 0., (), 0., 0., 0., 0., False, False
            for w in sum(p, ()):
                if not re.search('^\?', w):
                    p_tmp += (w, )
                    if re.search('\d+-\d+-\d+', w): w = w.split('-')[0] # When it's a datetime, only use year
                    if re.search('^[mg]\.', w): # When it's a mid
                        if w not in M2N: update_m2n(w, M2N)
                        if (w in M2N): path.append(M2N[w]) # 'e' #
                        topic_score += 0. if w not in te else te[w][0] if tokenizer.dataset in ['CQ'] else te[w] # if M2N[w] in batch.raw_question
                        #topic_number += 1.
                        if w in te:
                            topic_number += 1.
                        elif w in set(batch.const_type):
                            type_number += 1.
                        elif not contain_minimax:
                            skip = True # When it's a mid that doesn't belong to the question
                    elif len(w.split('.')) > 2:  # When it's a relation
                        path += w.split('.')[-1].split('_')
                    elif not w.isdigit(): # When it's superlative
                        if (batch.const_minimax is not None) and w in set(batch.const_minimax):
                            path.append(w); contain_minimax = True; superlative_number += 1.
                        else:
                            skip = True
                    elif (not contain_minimax): # When it's a year
                        if (batch.const_time is not None) and w in set(batch.const_time):
                            path.append(w); year_number += 1.
                        else:
                            skip = True
            if skip: continue
            if check_answer(raw_candidate_paths[p]) == 2: path.append('date')
            batch.candidate_paths += [p]
            path = tokenizer.tokenize(' '.join(path))
            #print(p, path)
            path = tokenizer.convert_tokens_to_ids(path)
            batch.candidate_answers += [clean_answer(raw_candidate_paths[p])]
            answer = np.log(len(raw_candidate_paths[p]))

            '''Append features for ranking'''
            candidate_paths += [path] # {path + [102] + answer, path}
            topic_scores += [topic_score]
            topic_numbers += [topic_number] #topic_number
            answer_numbers += [answer]
            type_numbers += [type_number]
            superlative_numbers += [superlative_number]
            year_numbers += [year_number]
            hop_numbers += [answer] # {0, answer}

            if tokenizer.dataset in ['CQ']:
                answer = []
                for w in raw_candidate_paths[p]:
                    if w not in M2N: update_m2n(w, M2N)
                    answer += [M2N[w].lower()]
            else:
                answer = clean_answer(raw_candidate_paths[p])

            # if (tokenizer.dataset in ['CWQ'] and time == 0): #I changed this! , 'WBQ'
            #     F1 = generate_Acc_tmp(p_tmp, batch.golden_graph[:len(p_tmp)] if time == 0 else batch.golden_graph)
            # else:
            F1 = generate_F1_tmp(answer, batch.answer)

            batch.history_candidate_paths.add((p, set(answer) == set(batch.answer)))
            batch.current_F1s += [F1]
            batch.orig_F1s += [F1]
            batch.F1s += [generate_F1_tmp(answer, batch.answer)] #  generate_Acc_tmp(p_tmp, batch.golden_graph[:len(p_tmp)])

            if len(path) > max_cp_length:
                max_cp_length = len(path) + len(answer) if isinstance(hop_numbers[-1], list) else len(path)

    #exit()
    '''Whether to stop or not'''
    batch.current_F1s, batch.F1s = np.array(batch.current_F1s), np.array(batch.F1s)
    if np.sum(batch.current_F1s) == 0: batch.current_F1s[:] = 1.
    if np.sum(batch.F1s) == 0: batch.F1s[:] = 1.
    batch.current_F1s /= np.sum(batch.current_F1s)
    batch.F1s /= np.sum(batch.F1s)
    stop = True if (len(candidate_paths) and time > 0) else False
    return candidate_paths, topic_scores, topic_numbers, type_numbers, superlative_numbers, year_numbers, answer_numbers, hop_numbers, RAs, max_cp_length, query_num, stop

def select_field(q, cp, ts, tn, ty_n, su_n, ye_n, an_n, hn, RAs, mcl, is_train=False, method='Bert', save_model='SO'):
    is_hn_list = isinstance(hn[0], list)
    if method in ['Bert']:
        mcl = np.min([mcl+len(q)+2+int(is_hn_list), 35])
        sequence, sequence_token, sequence_position = [], [], []
        for i in range(len(cp)):
            sequence += [[101] + q + [102] + cp[i]]
            sequence_token += [[0]*(len(q)+1) + [1]*(len(cp[i])+1)]  # I changed this !
            sequence_position += [list(range(len(q)+1)) + list(range(len(cp[i])+1))]
            if is_hn_list:
                sequence[-1] += ([102] + hn[i])
                sequence_token[-1] += ([2]*(len(hn[i])+1))
                sequence_position[-1] += list(range(len(hn[i])+1))

        cp = truncated_sequence(sequence, mcl)
        q = truncated_sequence(sequence_token, mcl)
        hn = truncated_sequence(sequence_position, mcl)
    else:
        q, cp = [q]*len(cp), truncated_sequence(cp, mcl)

    hn = torch.tensor(hn, dtype=torch.long).view(1, len(cp), -1)#
    q = torch.tensor(q, dtype=torch.long).view(1, len(cp), -1)
    cp = torch.tensor(cp, dtype=torch.long).view(1, len(cp), -1)#
    ts = torch.tensor(ts, dtype=torch.float).view(1, -1)#
    tn = torch.tensor(tn, dtype=torch.float).view(1, -1)
    ty_n = torch.tensor(ty_n, dtype=torch.float).view(1, -1)
    su_n = torch.tensor(su_n, dtype=torch.float).view(1, -1)
    ye_n = torch.tensor(ye_n, dtype=torch.float).view(1, -1)
    an_n = torch.tensor(an_n, dtype=torch.float).view(1, -1)
    #print(cp[0, :10, :]); print(q[0, :10, :]); print(hn[0, :10, :])#; exit()
    return q, cp, ts, tn, ty_n, su_n, ye_n, an_n, hn

def truncated_sequence(cp, mcl, fill=0):
    for c_idx, c in enumerate(cp):
        if len(c) > mcl:
            cp[c_idx] = c[:mcl]
        elif len(c) < mcl:
            cp[c_idx] += [fill] * (mcl - len(c))
    return cp

def select_action(policy, raw_logits, adjust_F1s = None, previous_action_num = None, is_train = True, k = 1, is_reinforce=True, time=0, dataset='WBQ', adjust_F1_weight=0.5):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    loss = None
    if is_train and is_reinforce == 2:
        '''If we use semi-supervision with beam'''
        logits = F.softmax(raw_logits, 1)
        if torch.isnan(logits).any(): print(logits[:10]); exit() # debug
        #print(adjust_F1s.nonzero()[:1, 1: 2])
        k = np.min([3, logits.size(1)])# if time <2 else np.min([100, logits.size(1)])
        #print(logits, adjust_F1s)
        #loss = nn.KLDivLoss(reduction='sum')(logits.log(), adjust_F1s)
        loss = nn.MSELoss()(logits, adjust_F1s)
        action = torch.argsort(adjust_F1s, dim =1, descending=True)[:, :k].view(1, k)
    elif is_train:
        '''If we use reinforcement learning'''
        #logits = relaxed_softmax(raw_logits)
        logits = F.softmax(raw_logits, 1)
        probs = (1.-adjust_F1_weight)*logits + adjust_F1_weight*adjust_F1s  # #
        if torch.isnan(probs).any(): print(probs[:10]); exit() # debug
        c = Categorical(probs=probs)
        action = c.sample((3, ))
        c_log_prob = torch.gather(probs.transpose(1, 0), 0, action)
        #print('train ', time, probs[:, :5], action)
        # Add log probability of our chosen action to our history
        if policy.policy_history.dim() != 0:
            policy.policy_history = torch.cat([policy.policy_history, c_log_prob.view(3, 1)], -1)
        else:
            policy.policy_history = c_log_prob.view(3, 1)

        # compute max-likelihood loss
        #if not is_reinforce:
            #loss = nn.KLDivLoss(reduction='sum')(logits.log(), F1s)
            #loss = nn.MSELoss()(logits, F1s)
    else:
        k = np.min([k, raw_logits.size(1)]) #if time <2 else np.min([100, raw_logits.size(1)])
        logits = adjust_F1s if adjust_F1s is not None else raw_logits
        #print('test time', logits[:, :5])
        action = torch.argsort(logits, dim =1, descending=True)[:, :k].view(k, 1)
    return action, loss

def update_train_instance(batch, action):
    batch.current_topic_entity, batch.previous_index, batch.candidate_path_index = {}, {}, []

    cp2pi = batch.candidate_paths2previous_index
    for a_idx, a in enumerate(action.reshape(-1)):
        top_answer = batch.candidate_answers[a]
        batch.previous_index[batch.candidate_paths[a]] = (a_idx, ) if len(cp2pi)== 0 else cp2pi[a] + (a_idx, )
        batch.candidate_path_index += [batch.candidate_paths[a]]
        for t in top_answer:
            batch.current_topic_entity[t] = batch.candidate_paths[a]

    batch.candidate_paths = []
    batch.candidate_answers = []
    batch.candidate_paths2previous_index = []
    batch.current_F1s = []
    batch.F1s = []
    batch.hop_number += 1

def generate_F1(logits, action, batch, time = 0, is_train = True, eval_metric = 'AnsAcc', M2N = None, top_pred_ans = None):
    if top_pred_ans is None or len(top_pred_ans) == 0: top_pred_ans = defaultdict(int)

    max_index  = np.argmax(logits)
    ca, ans, pred_cp, cp = batch.candidate_answers, batch.answer, None, batch.candidate_paths
    if not is_train:
        if eval_metric in ['Hits1']:
            for a_idx, a in enumerate(action.reshape(-1)):
                pred_ans = ca[a]
                #print(time, a_idx, cp[a], pred_ans)
                if str(cp[a]) not in top_pred_ans: top_pred_ans[str(cp[a])] = defaultdict(int)
                for an in pred_ans:
                    top_pred_ans[str(cp[a])][str(time)+str(a_idx)+ an] += logits[0, a]
        action = action[:1]
        pred_cp = ' '.join(sum(batch.candidate_paths[max_index], ()))

    F1s = []
    for a in action.reshape(-1):
        pred_ans = ca[a]
        if eval_metric in ['F1Text']:
            pred_ans = [M2N[w].lower() if w in M2N else w for w in pred_ans]
            F1 = generate_F1_tmp(pred_ans, ans)
        elif eval_metric in ['F1', 'Hits1']:
            F1 = generate_F1_tmp(pred_ans, ans)
        elif eval_metric in ['AnsAcc']:
            F1 = float(set(pred_ans) == set(ans))
        elif eval_metric in ['GraphAcc']:
            p_tmp = tuple([w for w in sum(cp[a], ()) if not re.search('^\?', w)])
            F1 = generate_Acc_tmp(p_tmp, batch.golden_graph[:len(p_tmp)])
        else:
            raise Exception('Evaluation metric is not correct !')
        F1s += [F1]

    stop = True if (max_index in np.arange(batch.previous_action_num) and time > 0) else False
    return F1s, pred_cp, stop, pred_ans, top_pred_ans

def generate_Acc_tmp(pred_graph, golden_graph):
    Acc = float(set(pred_graph) == set(golden_graph))
    return Acc

def generate_F1_tmp(pred_ans, ans):
    TP = len(set(pred_ans) & set(ans))
    precision = TP*1./np.max([len(set(pred_ans)), 1e-10])
    recall = TP*1./np.max([len(set(ans)), 1e-10])
    F1 = 2. * precision * recall/np.max([(precision + recall), 1e-10])
    return F1

def update_policy_immediately(adjust_loss, optimizer):
    # Update network weights
    optimizer.zero_grad()
    adjust_loss.backward()
    optimizer.step()

    return adjust_loss.item()

def update_policy(adjust_loss, policy, optimizer, batch, device = None, LM_loss =None, is_reinforce=None):
    previous_index, candidate_path_index = batch.previous_index, batch.candidate_path_index
    R = np.array([0] * len(policy.reward_episode[0]))
    raw_rewards = []

    # Discount future rewards back to the present using gamma
    for r_idx, r in enumerate(policy.reward_episode[::-1]):
        R = r + policy.gamma * R if r_idx == 0 else policy.gamma * R#np.array(r) +
        raw_rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(raw_rewards).transpose(1, 0)
    # rewards = rewards if len(raw_rewards) == 1 else (rewards-rewards.mean(-1).view(-1, 1)) / (rewards.std(-1).view(-1, 1)+np.finfo(np.float32).eps)
    #print(rewards)
    if device: rewards = Variable(rewards).to(device)

    # Calculate loss
    top_index = [previous_index[cp] for cp in candidate_path_index]
    top_index = torch.LongTensor(top_index)
    if device: top_index = Variable(top_index).to(device)
    if is_reinforce != 2: gather_policy = torch.gather(policy.policy_history, 0, top_index)
    loss = adjust_loss if adjust_loss is not None else torch.sum(torch.mul(gather_policy, rewards).mul(-1))
    if LM_loss: loss += LM_loss

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Save and intialize episode history counters
    # policy.loss_history.append(loss.item())
    # policy.reward_history.append(np.sum(policy.reward_episode))
    return loss.item()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_folder", default=None, type=str, help="QA folder for training. E.g., train")
    parser.add_argument("--dev_folder", default=None, type=str, help="QA folder for dev. E.g., dev")
    parser.add_argument("--test_folder", default=None, type=str, help="QA folder for test. E.g., test")
    parser.add_argument("--vocab_file", default=None, type=str, help="Vocab txt for vocabulary")
    parser.add_argument("--KB_file", default=None, type=str, help="KB json for question answering")
    parser.add_argument("--M2N_file", default=None, type=str, help="mid2name json for question answering")
    parser.add_argument("--QUERY_file", default=None, type=str, help="query json for recording searched queries")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written")

    # Other parameters
    parser.add_argument("--load_model", default=None, type=str, help="The pre-trained model to load")
    parser.add_argument("--save_model", default='BaseSave', type=str, help="The name that the models save as")
    parser.add_argument("--config", default='config/base_config.json', help="The config of base model")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="The epoches of training")
    parser.add_argument("--do_train", default=1, type=int, help="Whether to run training")
    parser.add_argument("--do_eval", default=1, type=int, help= "Whether to run eval")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Total batch size for training")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="Total batch size for eval")
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="Total number of training epoches to perform")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--seed", default=123, type=int, help="random seeed for initialization")
    parser.add_argument("--gpu_id", default=1, type=int, help="id of gpu")
    parser.add_argument("--top_k", default=1, type=int, help="retrieve top k relation path during prediction")
    parser.add_argument("--adjust_F1_weight", default=0.5, type=float, help="weight of adjust F1 score during training")
    parser.add_argument("--train_limit_number", default=150, type=int, help="the number of training instances")
    parser.add_argument("--max_hop_num", default=1, type=int, help="maximum hop number")
    parser.add_argument("--do_policy_gradient", default=1, type=int, help="Whether to train with policy gradient. 1: use policy gradient; 2: use maximum likelihood with beam")
    args = parser.parse_args()

    if torch.cuda.is_available():
        logger.info("cuda {} is available".format(args.gpu_id))
        device = torch.device("cuda", args.gpu_id) #
        n_gpu = 1
    else:
        device = None
        logger.info("cuda is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    load_model_file = args.load_model+".bin" if args.load_model else None
    save_model_file = os.path.join(args.output_dir, args.save_model+".bin") if args.save_model else os.path.join(args.output_dir, "base_model.bin")
    save_eval_cp_file = os.path.join(args.output_dir, args.save_model+"_predcp.txt")
    save_eval_file = os.path.join(args.output_dir, args.save_model+".txt")
    save_kb_cache = os.path.join(os.path.dirname(args.KB_file), "kb_cache.json")
    save_m2n_cache = os.path.join(os.path.dirname(args.M2N_file), "m2n_cache.json")
    save_query_cache = os.path.join(os.path.dirname(args.QUERY_file), "query_cache.json")

    tokenizer = Tokenizer(args.vocab_file)
    KB = {} if args.do_eval == 2 else convert_json_to_load(Load_KB_Files(args.KB_file)) if args.KB_file else None
    M2N = {} if args.do_eval == 2 else Load_KB_Files(args.M2N_file)
    QUERY = set() if args.do_eval == 2 else set(Load_KB_Files(args.QUERY_file))

    config = ModelConfig.from_json_file(args.config)
    policy = Policy(config, tokenizer.vocab, device)
    if load_model_file and os.path.exists(load_model_file):
        model_dic = torch.load(load_model_file, map_location='cpu')
        policy.load_state_dict(model_dic, strict=True)
        print("successfully load pre-trained model ...")
    elif config.method in ['Bert']:
        model_dic = torch.load('config/pytorch_model.bin', map_location='cpu')
        model_dic = {re.sub('bert', 'ranker', k): v for k, v in model_dic.items()}
        model_dic['ranker.embeddings.token_type_embeddings.weight'] = torch.cat([model_dic['ranker.embeddings.token_type_embeddings.weight'], model_dic['ranker.embeddings.token_type_embeddings.weight'][1:]], 0)
        if config.method in ['Bert_tmp']: model_dic.update({re.sub('encoder', 'KBencoder', k): v for k, v in model_dic.items() if re.search('encoder', k)})
        policy.load_state_dict(model_dic, strict=False)
        print("successfully load Bert model ...")
    else:
        print("successfully initialize model ...")
    #print(policy.ranker.decoder.weight.data); exit()
    if args.gpu_id:
        policy.to(device)

    global_step, max_eval_reward, t_total = 0, -0.1, 0
    if args.do_eval:
        dev_instances = create_instances(input_file=args.dev_folder,
                                          tokenizer=tokenizer)
        test_instances = create_instances(input_file=args.test_folder,
                                          tokenizer=tokenizer)
        logger.info("***** Loading evaluation *****")
        logger.info("   Num dev examples = %d", len(dev_instances))
        logger.info("   Num test examples = %d", len(test_instances))
        logger.info("   Batch size = %s", args.eval_batch_size)
    if args.do_train:
        train_instances = create_instances(input_file=args.train_folder,
                                           tokenizer=tokenizer)
        logger.info("***** Loading training ******")
        logger.info("    Num examples = %d" , len(train_instances))
        logger.info("    Batch size = %s", args.train_batch_size)
        t_total = len(train_instances)*args.num_train_epochs

    # Prepare optimizer
    # param_optimizer = list(policy.named_parameters())
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=t_total)
    param_optimizer = list(policy.parameters())
    optimizer = optim.Adam(param_optimizer, lr=args.learning_rate)
    #te_idx = json.load(open('data/train_CWQ/te_idx.json', 'r'))

    args.num_train_epochs = 1 if not args.do_train else args.num_train_epochs
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

        tr_loss, tr_LM_loss, tr_reward, tr_reward_boundary, hop1_tr_reward, nb_tr_examples, nb_tr_steps, query_num = 0., 0., 0., 0., 0, 0, 0, 0.
        if args.do_train:
            policy.train()
            if args.do_eval == 2: train_instances = train_instances[:1]
            random.shuffle(train_instances)
            #train_instances_ = [train_instances[t_idx] for t_idx in te_idx]
            #random.shuffle(train_instances_)

            for step, batch in enumerate(train_instances[:10000]): #train_instances_
                #print(step)
                done, skip_forward = False, False
                time, _total_losses = 0, 0

                while time < args.max_hop_num:
                    # Retrieve graphs based on the current graph
                    cp, ts, tn, ty_n, su_n, ye_n, an_n, hn, RAs, mcl, qr_n, done = retrieve_KB(batch, KB, QUERY, M2N, tokenizer,
                                    config.method, train_limit_number=args.train_limit_number, time = time, is_train=True,
                                    save_model=args.save_model)
                    query_num += qr_n

                    if len(cp) == 0: skip_forward = True; break # When there is no candidate paths for the question, skip
                    ready_batch = select_field(batch.question, cp, ts, tn, ty_n, su_n, ye_n, an_n, hn, RAs, mcl, is_train=True, method=config.method, save_model=args.save_model)
                    if args.gpu_id: ready_batch = tuple(t.to(device) for t in ready_batch)

                    # Step through environment using chosen action
                    _logits, _losses = policy(ready_batch, None)
                    _total_losses += _losses if _losses else 0
                    logits = _logits.cpu().data.numpy() if args.gpu_id else _logits.data.numpy()
                    adjust_F1s = torch.tensor(batch.current_F1s, dtype=torch.float).view(1, -1)
                    F1s = torch.tensor(batch.F1s, dtype=torch.float).view(1, -1)
                    if args.gpu_id: _adjust_F1s, _F1s = adjust_F1s.to(device), F1s.to(device)
                    if torch.isnan(_logits).any() or (_logits.size()!= _adjust_F1s.size()): skip_forward = True; break # When there is a bug, skip
                    _action, _adjust_loss = select_action(policy, _logits, adjust_F1s = _adjust_F1s,
                                                          previous_action_num = batch.previous_action_num,
                                                          is_train=True, time = time, is_reinforce=args.do_policy_gradient,
                                                          dataset=tokenizer.dataset, adjust_F1_weight = args.adjust_F1_weight) #True
                    if args.do_policy_gradient ==2: loss= update_policy_immediately(_adjust_loss, optimizer)
                    action = _action.cpu().data.numpy() if args.gpu_id else _action.data.numpy()
                    eval_metric = 'GraphAcc' if (time==0 and tokenizer.dataset in ['CWQ']) else 'AnsAcc' if (tokenizer.dataset in ['FBQ']) else 'F1Text' if (tokenizer.dataset in ['CQ']) else 'F1'
                    reward, _, done, _, _ = generate_F1(logits, action, batch, time = time, is_train=True, eval_metric=eval_metric, M2N=M2N)
                    if time== 0 and tokenizer.dataset in ['CWQ']: hop1_tr_reward += np.mean(reward)
                    update_train_instance(batch, action)

                    # Save reward
                    policy.reward_episode.append(reward)
                    if done: break # When the best path in the previous iteration is same as the best path in current iteration
                    time += 1
                #if np.max(batch.orig_F1s) > reward: print(np.max(batch.orig_F1s)); print(reward); exit()
                # Used to determine when the environment is solved.
                if not skip_forward:
                    if args.do_policy_gradient != 2:
                        lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        loss = update_policy(_adjust_loss, policy, optimizer, batch, device = device, LM_loss = _total_losses, is_reinforce=args.do_policy_gradient)

                    tr_loss += loss
                    if _total_losses: tr_LM_loss += _total_losses.item()
                    tr_reward_boundary += np.max(batch.orig_F1s)
                    tr_reward += np.mean(reward)
                    nb_tr_examples += 1
                    nb_tr_steps += 1
                    global_step += 1
                policy.reset()
                batch.reset()
                #print('max train reward', np.max(batch.orig_F1s), '\n')

                if (step + 1) % 5000 == 0:
                    print('trained %s instances ...' %step)
                    # model_to_save = policy.module if hasattr(policy, 'module') else policy
                    # torch.save(model_to_save.state_dict(), save_model_file)
                    # Save_KB_Files(convert_json_to_save(KB), save_kb_cache)
                    # Save_KB_Files(M2N, save_m2n_cache)
                    # Save_KB_Files(list(QUERY), save_query_cache)

        if args.do_eval:
            policy.eval()
            eval_reward, nb_eval_steps, nb_eval_examples = 0, 0, 0
            if args.do_eval == 2: dev_instances = dev_instances[:1]

            for eval_step, batch in enumerate(dev_instances):
                done, skip_forward, pred_cp = False, False, ''
                time = 0
                #print(eval_step)
                while time < args.max_hop_num:
                    time1 = mytime.time()
                    cp, ts, tn, ty_n, su_n, ye_n, an_n, hn, RAs, mcl, qr_n, done = retrieve_KB(batch, KB, QUERY, M2N, tokenizer, config.method, time = time)
                    query_num += qr_n

                    if len(cp) == 0: skip_forward = True; break
                    ready_batch = select_field(batch.question, cp, ts, tn, ty_n, su_n, ye_n, an_n, hn, RAs, mcl, method=config.method)
                    if args.gpu_id: ready_batch = tuple(t.to(device) for t in ready_batch)

                    # Step through environment using chosen action
                    with torch.no_grad():
                        _logits, _ = policy(ready_batch, None)
                    logits = _logits.cpu().data.numpy() if args.gpu_id else _logits.data.numpy()

                    _action, _ = select_action(policy, _logits, is_train=False, k=args.top_k, dataset=tokenizer.dataset)
                    action = _action.cpu().data.numpy() if args.gpu_id else _action.data.numpy()
                    eval_metric = 'AnsAcc' if (tokenizer.dataset in ['FBQ']) else 'F1Text' if (tokenizer.dataset in ['CQ']) else 'F1'
                    reward, pred_cp, done, _, _ = generate_F1(logits, action, batch, time = time, is_train = False, eval_metric=eval_metric, M2N=M2N)
                    update_train_instance(batch, action)

                    if done: break
                    time += 1

                if not skip_forward:
                    eval_reward += np.mean(reward)
                    nb_eval_examples += 1
                    nb_eval_steps += 1
                batch.reset()
                #print(logits); exit()
            result = {'training loss': tr_loss/np.max([nb_tr_examples, 1.e-10]),
                      'training reward': tr_reward/np.max([nb_tr_examples, 1.e-10]),
                      'dev reward': eval_reward/np.max([nb_eval_examples, 1.e-10])}
            if tokenizer.dataset in ['CWQ', 'WBQ', 'CQ']: result['train reward boundary'] = tr_reward_boundary/np.max([nb_tr_examples, 1.e-10])
            if tokenizer.dataset in ['CWQ']: result['training hop1 acc'] = hop1_tr_reward/np.max([nb_tr_examples, 1.e-10])
            if 'LM' in config.method: result['training LM loss'] = tr_LM_loss/np.max([nb_tr_examples, 1.e-10])
            eval_reward = eval_reward/np.max([nb_eval_examples, 1.e-10])

            if eval_reward >= max_eval_reward:
                max_eval_reward = eval_reward
                if args.do_eval == 2: test_instances = test_instances[:1]
                eval_reward, nb_eval_steps, nb_eval_examples, eval_pred_cps, eval_pred_top_ans, eval_reward_boundary = 0, 0, 0, [], [], 0

                for eval_step, batch in enumerate(test_instances): #[328:329]
                    done, skip_forward, pred_cp = False, False, ''
                    time, reward, top_pred_ans = 0, [0], defaultdict(int)
                    #print(eval_step)
                    while time < args.max_hop_num:
                        time1 = mytime.time()
                        cp, ts, tn, ty_n, su_n, ye_n, an_n, hn, RAs, mcl, qr_n, done = retrieve_KB(batch, KB, QUERY, M2N, tokenizer, config.method, time = time)
                        query_num += qr_n

                        if len(cp) == 0:
                            skip_forward = True
                            break
                        ready_batch = select_field(batch.question, cp, ts, tn, ty_n, su_n, ye_n, an_n, hn, RAs, mcl, method=config.method)
                        if args.gpu_id: ready_batch = tuple(t.to(device) for t in ready_batch)

                        # Step through environment using chosen action
                        with torch.no_grad():
                            _logits, _ = policy(ready_batch, None)
                            _logits = F.softmax(_logits, 1)
                        logits = _logits.cpu().data.numpy() if args.gpu_id else _logits.data.numpy()
                        adjust_F1s = torch.tensor(batch.current_F1s, dtype=torch.float).view(1, -1)
                        if args.gpu_id: _adjust_F1s = adjust_F1s.to(device)

                        _action, _ = select_action(policy, _logits, is_train=False, k=args.top_k, dataset=tokenizer.dataset) # adjust_F1s = _adjust_F1s,  if time < 2 else None
                        action = _action.cpu().data.numpy() if args.gpu_id else _action.data.numpy()
                        eval_metric = 'AnsAcc' if (tokenizer.dataset in ['FBQ']) else 'F1Text' if (tokenizer.dataset in ['CQ']) else 'Hits1' if (tokenizer.dataset in ['CWQ']) else 'F1'
                        reward, pred_cp, done, pred_ans, top_pred_ans = generate_F1(logits, action, batch, time = time, is_train = False, eval_metric=eval_metric, M2N=M2N, top_pred_ans=top_pred_ans)
                        update_train_instance(batch, action)
                        if done: break
                        time += 1
                    #if len(pred_cp.split(' ')) < 2: print(eval_step); exit()
                    eval_pred_cps += [re.sub('\n', '', '%s\t%s\t%s\t%s' %(eval_step+1, pred_cp, reward, '\t'.join(pred_ans)))]
                    eval_pred_top_ans += [top_pred_ans]
                    #print(top_pred_ans)

                    if not skip_forward:
                        #if np.max(batch.orig_F1s) > np.mean(reward): print(batch.orig_F1s); print(reward); print(eval_step); exit()
                        eval_reward += np.mean(reward)
                        eval_reward_boundary += np.max(batch.orig_F1s)
                        nb_eval_examples += 1
                        nb_eval_steps += 1
                    batch.reset()
                    #print('max reward', np.max(batch.orig_F1s), '\n')

                result['test reward'] = eval_reward/np.max([nb_eval_examples, 1.e-10])
                result['query times'] = '%s (save model) ' %(query_num)
                if args.do_eval == 2: print(result); exit()
                if tokenizer.dataset in ['CWQ', 'WBQ', 'CQ']: result['test reward boundary'] = eval_reward_boundary/np.max([nb_eval_examples, 1.e-10])
                g = open(save_eval_cp_file, "w")
                g.write('\n'.join(eval_pred_cps))
                g.close()
                if eval_pred_top_ans:
                    g = open(re.sub('.txt$', '.json', save_eval_cp_file), "w")
                    for top_pred_ans in eval_pred_top_ans:
                        json.dump(top_pred_ans, g)
                        g.write('\n')
                    g.close()

                if args.do_train:
                    '''save the model and some kb cache'''
                    model_to_save = policy.module if hasattr(policy, 'module') else policy
                    torch.save(model_to_save.state_dict(), save_model_file)
                    Save_KB_Files(convert_json_to_save(KB), save_kb_cache) #KB_save
                    Save_KB_Files(M2N, save_m2n_cache)
                    Save_KB_Files(list(QUERY), save_query_cache) #QUERY_save

            with open(save_eval_file, "a") as writer:
                logger.info("***** Eval results (%s)*****" %epoch)
                writer.write("***** Eval results (%s)*****\n" %epoch)
                for key in sorted(result.keys()):
                    logger.info(" %s=%s", key, str(result[key]))
                    writer.write("%s=%s \n" %(key, str(result[key])))
            #exit()
if __name__ == '__main__':
    main()
