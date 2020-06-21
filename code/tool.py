import json
import os
from os import listdir
from SPARQL_test import *
import numpy as np
import sqlite3
import time
from ast import literal_eval
#import redis
import random

np.random.seed(123)

def update_hierarchical_dic_with_unit(kb, head, r, t):
    if head not in kb:
        kb[head] = {r: set([t])}
    elif r not in kb[head]:
        kb[head][r] = set([t])
    else:
        kb[head][r].add(t)

def update_m2n(e, m2n):
    name = SQL_entity2name(e)
    m2n[e] = name

def update_hierarchical_dic_with_set(dic, x, y, z):
    if x not in dic:
        dic[x] = {y: z}
    elif y not in dic[x]:
        dic[x][y] = z
    else:
    	dic[x][y].update(z)

def convert_json_to_save(dic):
    '''If dictionary is {keys1(tuple): keys2(tuple): value(set)}'''
    new_dic = {}
    for hs in dic:
        new_h = '\t'.join([' '.join(r) for r in hs])
        if len(dic[hs]) == 0: new_dic[new_h] = {}
        for rs in dic[hs]:
            new_r = '\t'.join([' '.join(r) for r in rs])
            new_t = list(dic[hs][rs])
            if new_h not in new_dic:
                new_dic[new_h] = {new_r: new_t}
            elif new_r not in new_dic[new_h]:
                new_dic[new_h][new_r] = new_t
    return new_dic

def convert_json_to_load(dic):
    new_dic = {}
    for hs in dic:
        new_h = tuple([tuple(r.split(' ')) for r in hs.split('\t')])
        if len(dic[hs]) == 0: new_dic[new_h] = {}
        for rs in dic[hs]:
            new_r = tuple([tuple(r.split(' ')) for r in rs.split('\t')])
            new_t = set(dic[hs][rs])
            if new_h not in new_dic:
                new_dic[new_h] = {new_r: new_t}
            elif new_r not in new_dic[new_h]:
                new_dic[new_h][new_r] = new_t
    return new_dic

def convert_multiple_to_one(data_path):
    kb = {}
    m2n = {}
    query = set()
    for filename in listdir(data_path):
        with open(os.path.join(data_path, filename)) as f:
            if 'kb' in filename:
                kb_tmp = convert_json_to_load(json.load(f))
                for te in kb_tmp:
                    for trip in kb_tmp[te]:
                        if len(trip[0]) == 3:
                            update_hierarchical_dic_with_set(kb, te, trip, kb_tmp[te][trip])
                        else:
                            print(trip); exit()
            elif 'm2n' in filename:
                m2n.update(json.load(f))
            elif 'query' in filename:
                query.update(set(json.load(f)))

    g = open(os.path.join(data_path, 'm2n.json'), 'w')
    json.dump(m2n, g)
    g.close()
    g = open(os.path.join(data_path, 'kb.json'), 'w')
    json.dump(convert_json_to_save(kb), g)
    g.close()
    g = open(os.path.join(data_path, 'query.json'), 'w')
    json.dump(query, g)
    g.close()

def convert_kbdic_to_db(data_path):
    kb = convert_json_to_load(json.load(open(os.path.join(data_path, 'kb_cache.json'))))
    conn = sqlite3.connect(os.path.join(data_path, 'kb_cache.db'))
    c = conn.cursor()

    # Create table
    c.execute('''Create table kb (query, triplet, answer)''')
    # Insert a row of data
    for query in kb:
        if len(kb[query]) == 0:
            c.execute('''Insert into kb values ("{0}", "", "")'''.format(str(query)))
        for triplet in kb[query]:
            answer = str(list(kb[query][triplet])).replace('"', '\'')
            c.execute('''Insert into kb values ("{0}", "{1}", "{2}")'''.format(str(query), str(triplet), answer))

    conn.commit()
    conn.close()

def convert_db_to_load(data_path):
    conn = sqlite3.connect(data_path)
    c = conn.cursor()
    return c, conn

def execute_db(c, query):
    results = c.execute('''Select * From kb Where query=?''', (str(query),))
    kb = {}
    for _, triplet, answer in results:
        if len(triplet) == 0: continue
        answer = set(answer[2: -2].split("', '"))
        kb[literal_eval(triplet)] = answer
    return kb

def convert_db_to_save(c, conn, kb):
    try:
        for query in kb:
            if len(kb[query]) == 0 and len(c.execute('''Select * From kb Where query=?''', (str(query),))) == 0:
                c.execute('''Insert into kb values ("{0}", "", "")'''.format(str(query)))
                continue
            for triplet in kb[query]:
                if len(c.execute('''Select * From kb Where triplet=?''', (str(triplet),))) == 0:
                    answer = str(list(kb[query][triplet])).replace('"', '\'')
                    c.execute('''Insert into kb values ("{0}", "{1}", "{2}")'''.format(str(query), str(triplet), answer))
        conn.commit()
        kb = {}
    except:
        print('SQL is busy! Save next time ...')
    return kb

def test_db(data_path):
    conn = sqlite3.connect(os.path.join(data_path, 'kb_cache.db'))
    c = conn.cursor()
    time1 = time.time()
    results = c.execute('''Select * From kb Where query=?''', ("(('m.0d05w3', ), )",))
    # kb = {}
    # for _, triplet, answer in results:
    #     if len(triplet) == 0: continue
    #     answer = set(answer[2: -2].split("', '"))
    #     kb[literal_eval(triplet)] = answer
    print(time.time() - time1)

def test_dic(data_path):
    kb = convert_json_to_load(json.load(open(os.path.join(data_path, 'kb_cache.json'))))
    r = redis.StrictRedis('10.0.104.57', port=6379)
    for query in kb:
        for triplet in kb[query]:
            r.set(str(query), str(triplet))
    time1 = time.time()
    r.get("(('m.0d05w3', ), )")
    print(time.time() - time1)

def generate_longest_string(X, Y):
    m, n = len(X), len(Y)
    L = [[None]*(n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    return L[m][n]

def generate_overlap_idx(q, mid):
    mid = mid.split()
    q = q.split()
    longest_string, longest_string_boundary = 0, (0, 0)
    for i in range(len(q) - len(mid) + 1):
        longest_string_len = generate_longest_string(' '.join(mid), ' '.join(q[i: i + len(mid)]))
        if longest_string_len > longest_string:
            longest_string_boundary = (i, i + len(mid))
            longest_string = longest_string_len
    return longest_string_boundary

def merge(lsts):
    sts = [set(np.arange(l[0], l[1]+1)) for l in lsts]
    i = 0
    while i < len(sts):
        j = i+1
        while j < len(sts):
            if len(sts[i].intersection(sts[j])) > 0:
                sts[i] = sts[i].union(sts[j])
                sts.pop(j)
            else: j += 1                        #---corrected
        i += 1
    lst = [(np.min(list(s)), np.max(list(s))) for s in sts]
    return lst

def generate_template(q, mids):
    overlap_idx = set()
    for mid in mids:
        overlap_idx.add(generate_overlap_idx(q, mid))
    overlap_idx = merge(overlap_idx)
    q_split = q.split()
    replace_str = []
    for s, e in overlap_idx:
        replace_str += [' '.join(q_split[s:e])]
    q = re.sub('|'.join(replace_str), 'E', q)

    return q

def generate_evaluation_tmp(pred_ans, ans):
    if len(pred_ans) == 0: return 0., 0., 0., 0., 0
    if ans == set(['']): return 1., 1., 1., 1., 1.
    TP = len(pred_ans & ans)
    acc = int(pred_ans == ans)
    precision = TP*1./np.max([len(pred_ans), 1e-10])
    recall = TP*1./np.max([len(ans), 1e-10])
    F1 = 2. * precision * recall/np.max([(precision + recall), 1e-10])
    one_ans = random.sample(pred_ans, 1)[0]
    hit1 = int(one_ans in ans)
    return acc, precision, recall, F1, hit1


def convert_sparql2graph(data_path):
    # q2sparql = {}
    # with open('/home/yunshi/Documents/Experiment/data/WebQSP/data/WebQSP.train.json') as f:
    #     lines = json.load(f)
    #     for line_idx, line in enumerate(lines['Questions']):
    #         q2sparql[line['ProcessedQuestion']] = line['Parses'][0]['Sparql']
    #
    # g = open(os.path.join(data_path, 'sparql.json'), 'w')
    # with open(os.path.join(data_path, 'q.txt')) as f:
    #     for line_idx, line in enumerate(f):
    #         line = line.strip()
    #         json.dump({'Sparql': q2sparql[line]}, g)
    #         g.write('\n')
    # g.close()
    # exit()
    new_sparqls, empty_count = [], 0
    with open(os.path.join(data_path, 'sparql.json')) as f1, \
        open(os.path.join(data_path, 'g.txt')) as f2:
        for line_idx, line in enumerate(zip(f1, f2)):
            line, line2 = line
            line2 = line2.strip()
            line = json.loads(line.strip())
            sparql = line['Sparql']
            sparql = re.split('\)\n(?=ns)|[ ]*\.\n[\t]*|\{\n[\t]*(?=ns)', sparql)
            select_sparql = [s for s in sparql if not (('PREFIX' in s) or ('langMatches' in s) or ('?' not in s) or ('!=' in s))]

            t, v_set, new_sparql, skip, display = 1, {}, [], False, False
            if line_idx in [692]: del select_sparql[2]
            for s_idx, s in enumerate(select_sparql):
                if '\nEXISTS' in s:
                    s = re.findall('(?<=\nEXISTS \{)[^\n]+', s)
                    s = s[-1][:-3]
                if '}\nORDER BY' in s or 'FILTER (str(?sk0) = ' in s:
                    continue
                if not skip:
                    for v in s.split()[:3]:
                        if re.search('^ns\:', v):
                            v = re.sub('^ns\:', '', v)
                        elif v == '?y':
                            if v not in v_set: v_set[v] = '?d%s' %t
                        elif v == '?x':
                            if v not in v_set: v_set[v] = '?e%s' %t
                        elif v == '?sk0':
                            if '}\nORDER BY' in select_sparql[s_idx + 1]:
                                time = test_sk0(re.sub('SELECT DISTINCT \?x', 'SELECT DISTINCT ?x, ?sk0', line['Sparql']))
                                superlative = 'last' if 'DESC' in select_sparql[s_idx + 1] else 'first'
                                new_sparql.insert(-2, superlative)
                                if v not in v_set: v_set[v] = time
                            elif 'FILTER (str(?sk0) = ' in select_sparql[s_idx + 1]:
                                if v not in v_set: v_set[v] = select_sparql[s_idx + 1].split('\"')[1]
                            else:
                                print(select_sparql)
                        elif '?sk' in v:
                            time = re.findall('[\d]+-[\d]+-[\d]+', line['Sparql'])[0].split('-')[0]
                            if v not in v_set: v_set[v] = time
                            display = True
                        else:
                            print(select_sparql); skip=True; break
                        new_sparql += [v_set[v] if v in v_set else v]
                t = len(v_set) + 1
            #if skip: print(line['Sparql'])

            if not skip:
                try:
                    new_sparqls += [' '.join(new_sparql)]
                except:
                    line2 = line2.split('\t')
                    if len(line2) == 1:
                        new_sparql = line2
                    elif len(line2) == 2:
                        new_sparql = line2 + ['?e1']
                    elif len(line2) == 3:
                        new_sparql = line2[:2] + ['?d1'] + ['?d1'] + line2[2:3] + ['?e2']
                    new_sparqls += [' '.join(new_sparql)]
            else:
                line2 = line2.split('\t')
                if len(line2) == 1:
                    new_sparql = line2
                elif len(line2) == 2:
                    new_sparql = line2 + ['?e1']
                elif len(line2) == 3:
                    new_sparql = line2[:2] + ['?d1'] + ['?d1'] + line2[2:3] + ['?e2']
                print(new_sparqls)
                new_sparqls += [' '.join(new_sparql)]

    print(empty_count)
    g = open(os.path.join(data_path, 'g_tmp.txt'), 'w')
    g.write('\n'.join(new_sparqls))
    g.close()

if __name__ == '__main__':
    #convert_kbdic_to_db('data/FBQ')
    convert_sparql2graph('data/train_WBQ')
    #execute_dic('data/FBQ')
