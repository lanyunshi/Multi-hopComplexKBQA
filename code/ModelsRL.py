import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from BertModel import *

import math
import numpy as np
import json
import zipfile
import os
import copy
import gc

eps = np.finfo(np.float32).eps.item()

class ModelConfig(object):
    """Configuration class to store the configuration of a 'Model'
    """
    def __init__(self,
                vocab_size_or_config_json_file,
                hidden_size = 200,
                dropout_prob = 0.1,
                initializer_range= 0.02):

        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.dropout_prob = dropout_prob
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """COnstruct a 'Config' from a Python dictionary of parameters."""
        config = ModelConfig(vocab_size_or_config_json_file = -1)
        for key, value in json_object.items():
            config.__dict__[key]=value
        return config
    @classmethod
    def from_json_file(cls, json_file):
        """Construct a 'Config' from a json file of parameters"""
        with open(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class SimpleRelRanker(nn.Module):
    """Construct the relation ranker for each step"""
    def __init__(self, config, device = None):
        super(SimpleRelRanker, self).__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.bidirectional = config.bidirectional
        self.num_layers = config.num_layers
        #self.encoder = nn.LSTM(self.hidden_size, self.hidden_size)
        self.q_encoder = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, bidirectional=bool(self.bidirectional))
        self.s_encoder = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, bidirectional=bool(self.bidirectional))
        self.decoder = nn.Linear(7, 1)

    def forward(self, batch, golden_ids=None):
        question, subgraph, ts_score, ts_nb, ty_nb, su_nb, ye_nb, an_nb, hop_nb = batch
        question = question[:, 0, :]
        b, ql = question.size()
        b, gn, gl = subgraph.size()
        question_emb = self.word_embeddings(question).transpose(1, 0) # (batch, qlen) --> (batch, qlen, hidden_size)
        subgraph_emb = self.word_embeddings(subgraph).view(b*gn, gl, self.hidden_size).transpose(1, 0) # (batch, gnum, glen) --> (batch, gnum, glen, hidden_size)

        question_emb, _ = self.q_encoder(question_emb)
        subgraph_emb, _ = self.s_encoder(subgraph_emb)

        question_emb = question_emb.view(ql, b, (self.bidirectional+1), self.hidden_size)
        subgraph_emb = subgraph_emb.view(gl, b*gn, (self.bidirectional+1), self.hidden_size)
        if self.bidirectional:
            question_rep = torch.cat([question_emb[-1:, :, 0, :], question_emb[:1, :, 1, :]], dim = 2)
            subgraph_rep = torch.cat([subgraph_emb[-1:, :, 0, :], subgraph_emb[:1, :, 1, :]], dim = 2)
        else:
            question_rep = question_emb[-1:]
            subgraph_rep = subgraph_emb[-1:]
        question_rep = question_rep.view(b, 1, (self.bidirectional+1)*self.hidden_size) # (batch, qlen, hidden_size) --> (batch, hidden_size)
        subgraph_rep = subgraph_rep.view(b, gn, (self.bidirectional+1)*self.hidden_size) # (batch, gnum, glen, hidden_size) --> (batch, gnum, hidden_size)

        sequence_sim = torch.bmm(subgraph_rep, question_rep.transpose(2, 1))
        features = torch.cat([ts_score.view(b, gn, 1), ts_nb.view(b, gn, 1), ty_nb.view(b, gn, 1), \
                              su_nb.view(b, gn, 1), ye_nb.view(b, gn, 1), an_nb.view(b, gn, 1), sequence_sim], 2) # hop_nb.view(b, gn, 1)
        logits = self.decoder(features).view(b, gn)

        return logits, 0

class MatchAggregationRanker(nn.Module):
    """Construct a 'Match-Aggregation' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(MatchAggregationRanker, self).__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.pre_question_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), bidirectional=True, batch_first=True)
        # self.pre_subgraph_encoder = nn.LSTM(self.hidden_size, int(self.hidden_size/2), bidirectional=True, batch_first=True)
        self.question_encoder = nn.LSTM(2*self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.subgraph_encoder = nn.LSTM(2*self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.sim = nn.Linear(4*self.hidden_size, 1)
        self.decoder = nn.Linear(7, 1)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.device = device
        self.padding_id = 0

    def forward(self, batch, golden_ids=None):
        question, subgraph, ts_score, ts_nb, ty_nb, su_nb, ye_nb, an_nb, hop_nb = batch
        question = question[:, 0, :]
        b, ql = question.size()
        b, gn, gl = subgraph.size()
        #print(b, ql, gn, gl)
        question_emb = self.word_embeddings(question) # (batch, qlen) --> (batch, qlen, hidden_size)
        subgraph_emb = self.word_embeddings(subgraph) # (batch, gnum, glen) --> (batch, gnum, glen, hidden_size)
        # question_emb = self.pre_question_encoder(question_emb)[0].view(b, ql, self.hidden_size)
        # subgraph_emb = self.pre_subgraph_encoder(subgraph_emb.view(b*gn, gl, -1))[0].view(b, gn, gl, self.hidden_size)
        question_emb = self.dropout(question_emb)
        subgraph_emb = self.dropout(subgraph_emb)

        question_mask = 1 - torch.eq(question, self.padding_id).type(torch.FloatTensor).view(b, ql, 1)
        subgraph_mask = 1 - torch.eq(subgraph, self.padding_id).type(torch.FloatTensor).view(b, gn*gl, 1)
        mask = torch.bmm(subgraph_mask, question_mask.transpose(2, 1)) # (batch, gnum*glen, 1) * (batch, 1, qlen) --> (batch, gnum*glen, qlen)
        mask = mask.to(self.device) if self.device else mask
        #print(subgraph_emb.size(), question_emb.size())
        attention = torch.bmm(subgraph_emb.contiguous().view(b, gn*gl, self.hidden_size), question_emb.transpose(2, 1)) # (batch, gnum*glen, hidden_size) * (batch, hidden_size, qlen) --> (batch, gnum*glen, qlen)
        mask_value = -1.e10 * torch.ones_like(mask).to(self.device) if self.device else -1.e10 * torch.ones_like(mask)
        attention = (mask * attention + (1 - mask) * mask_value).view(b, gn, gl, ql) # (batch, gnum, glen, qlen)
        atten_question = F.softmax(attention.view(b, gn*gl, ql), 2) # attentions along questions
        align_question = torch.bmm(atten_question, question_emb).view(b*gn, gl, self.hidden_size) # (batch, gnum*glen, hidden_size)
        atten_subgraph = F.softmax(attention, 2).view(b*gn, gl, ql) # attention along subgraphs
        align_subgraph = torch.bmm(atten_subgraph.transpose(2, 1), subgraph_emb.view(b*gn, gl, self.hidden_size)).view(b*gn, ql, self.hidden_size)

        compa_question = torch.max(self.question_encoder(torch.cat([align_question, subgraph_emb.view(b*gn, gl, self.hidden_size)], 2))[0], 1)[0] # (batch*gnum, 2*hidden_size)
        compa_subgraph = torch.max(self.subgraph_encoder(torch.cat([align_subgraph, question_emb.unsqueeze(1).repeat(1, gn, 1, 1).view(b*gn, ql, self.hidden_size)], 2))[0], 1)[0] # (batch*gnum, 2*hidden_size)
        # compa_question = torch.mean(self.question_encoder(torch.cat([align_question, subgraph_emb.view(b*gn, gl, self.hidden_size)], 2))[0], 1) # (batch*gnum, 2*hidden_size)
        # compa_subgraph = torch.mean(self.subgraph_encoder(torch.cat([align_subgraph, question_emb.unsqueeze(1).repeat(1, gn, 1, 1).view(b*gn, ql, self.hidden_size)], 2))[0], 1) # (batch*gnum, 2*hidden_size)
        sequence_sim = self.sim(torch.cat([compa_question, compa_subgraph], 1)).view(b, gn, 1) #F.cosine_similarity(compa_question, compa_subgraph).view(b, gn, 1)

        features = torch.cat([ts_score.view(b, gn, 1), ts_nb.view(b, gn, 1), ty_nb.view(b, gn, 1), \
                              su_nb.view(b, gn, 1), ye_nb.view(b, gn, 1), an_nb.view(b, gn, 1), sequence_sim], 2) # hop_nb.view(b, gn, 1)
        logits = self.decoder(features).view(b, gn)

        return logits, 0

class BertRanker(nn.Module):
    """Construct a 'BERT' model to rank relations for each step"""
    def __init__(self, config, device = None):
        super(BertRanker, self).__init__()
        self.hidden_size = config.hidden_size
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.sim = nn.Linear(self.hidden_size, 1)
        self.decoder = nn.Linear(7, 1)
        self.device = device
        self.padding_id = 0
        self.cls_id = 101
        self.sep_id = 102

    def forward(self, batch, golden_ids=None):
        sequence_token, sequence, ts_score, ts_nb, ty_nb, ye_nb, su_nb, an_nb, sequence_position = batch
        b, gn, sl = sequence.size()
        #print(b, gn, sl)

        sequence = sequence.view(b*gn, sl) # (batch*gnum, glen)
        sequence_token = sequence_token.view(b*gn, sl) # , hop_nb_token
        #print(sequence[:5, :20]); print(sequence_token[:5, :20]); exit()
        #sequence_position = sequence_position.view(b*gn, sl)

        sequence_mask = torch.eq(sequence, self.padding_id).type(torch.FloatTensor)
        sequence_mask = sequence_mask.to(self.device) if self.device else sequence_mask
        sequence_mask = sequence_mask.unsqueeze(1).unsqueeze(2)
        sequence_mask = -1.e10 * sequence_mask

        sequence_emb = self.embeddings(sequence, token_ids=sequence_token) # (batch, slen, hidden_size)
        sequence_enc = self.encoder(sequence_emb,
                                    sequence_mask,
                                    output_all_encoded_layers = False)[-1]
        sequence_out = sequence_enc
        sequence_pool = self.pooler(sequence_out) # (batch+batch*gnum, hidden_size)

        sequence_pool = self.dropout(sequence_pool.view(b, gn, self.hidden_size))
        sequence_sim = self.sim(sequence_pool) # hop_nb.view(b, gn, 1)
        features = torch.cat([ts_score.view(b, gn, 1), ts_nb.view(b, gn, 1), ty_nb.view(b, gn, 1), \
                              su_nb.view(b, gn, 1), ye_nb.view(b, gn, 1), an_nb.view(b, gn, 1), sequence_sim], 2) # (batch, gnum, 2*hidden_size+2) su_nb.view(b, gn, 1)
        #features = self.dropout(features)
        #print(features)

        logits = self.decoder(features).view(b, gn)
        #print(logits)

        return logits, 0

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, vocab, *input, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config
        self.vocab = vocab

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.Embedding)) and self.config.hidden_size==300 and module.weight.data.size(0) > 1000:
            if os.path.exists(self.config.Word2vec_path):
                embedding = np.load(self.config.Word2vec_path)
                module.weight.data = torch.tensor(embedding, dtype=torch.float)
                print('pretrained GloVe embeddings')
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                gloves = zipfile.ZipFile('/home/yunshi/Word2vec/glove.840B.300d.zip')
                seen = 0

                for glove in gloves.infolist():
                    with gloves.open(glove) as f:
                        for line in f:
                            if line != "":
                                splitline = line.split()
                                word = splitline[0].decode('utf-8')
                                embedding = splitline[1:]

                            if word in self.vocab and len(embedding) == 300:
                                temp = np.array([float(val) for val in embedding])
                                module.weight.data[self.vocab[word], :] = torch.tensor(temp, dtype=torch.float)
                                seen += 1

                print('pretrianed vocab %s among %s' %(seen, len(self.vocab)))
                np.save(self.config.Word2vec_path, module.weight.data.numpy())
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class Policy(PreTrainedModel):
    def __init__(self, config, vocab, device):
        super(Policy, self).__init__(config, vocab, device=None)
        self.device = device
        if config.method == "Siamese":
            self.ranker = SimpleRelRanker(config, device=device)
        elif config.method == "MatchAggregation":
            self.ranker = MatchAggregationRanker(config, device=device)
        elif config.method == "Bert":
            self.ranker = BertRanker(config, device=device)
        self.apply(self.init_bert_weights)

        self.gamma = config.gamma

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor()).to(self.device) if self.device else Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x, mg = None, chunk_size = 30):
        gn = x[1].size(1)
        if gn > chunk_size:
            chunk, logits, losses = int(np.ceil(gn/chunk_size)), [], 0
            mg_chunk_size = int(np.ceil(mg[0].size(1)/chunk)) if mg is not None else None
            for i in range(chunk):
                new_x = tuple([xx[:, i*chunk_size: (i+1)*chunk_size] for xx in x])
                #print(new_x[0].size())
                new_mg = tuple([xx[:, i*mg_chunk_size: (i+1)*mg_chunk_size] for xx in mg]) if mg is not None else mg
                logit, loss = self.ranker(new_x, new_mg)
                logits += [logit]
                losses += loss
            logits = torch.cat(logits, 1)
        else:
            logits, losses = self.ranker(x, mg)
        return logits, losses

    def reset(self):
        self.policy_history = Variable(torch.Tensor()).to(self.device) if self.device else Variable(torch.Tensor())
        self.reward_episode= []
