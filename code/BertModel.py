import copy
import torch
from torch import nn
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings"""
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_ids is None:
            token_ids = torch.zeros_like(input_ids)

        # print(input_ids[0, :])
        # print(token_ids[0, :])
        # print(position_ids[0, :])
        # exit()
        word_embeddings = self.word_embeddings(input_ids)
        token_embeddings = self.token_type_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings+ token_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mode='FirstPool'):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        if mode == 'None':
            first_token_tensor = hidden_states
        elif mode == 'FirstPool':
            first_token_tensor = hidden_states[:, 0]
        elif mode == 'MaxPool':
            first_token_tensor, _ = torch.max(hidden_states, 1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # print('\n')
        # for v in attention_probs.cpu().numpy()[0, :, :, :]:
        #     print(','.join(map(str, v)))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class LSTMCellWithProjection(torch.nn.Module):
    """AN LSTM with Recurrent Dropout and a projected and clipped hidden state and memory."""
    def __init__(self, config, go_forward):
        super(LSTMCellWithProjection, self).__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.cell_size = config.cell_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.go_forward = go_forward

        self.state_projection_clip_value = config.state_projection_clip_value
        self.memory_cell_clip_value = config.memory_cell_clip_value
        self.recurrent_dropout_probability = config.recurrent_dropout_probability

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.input_linearity = torch.nn.Linear(self.input_size, 4*self.cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(self.hidden_size, 4*self.cell_size, bias=True)

        self.state_projection = torch.nn.Linear(self.cell_size, self.hidden_size, bias=False)

    def forward(self, input_ids, batch_lengths, initial_state=None):
        batch_size, total_timesteps, _ = input_ids.size() # (batch_size, length, input_size)
        output_accumulator = torch.zeros((batch_size, total_timesteps, self.hidden_size), device=input_ids.device) # (batch_size, legnth, hidden_size)

        if initial_state is None:
            full_batch_previous_memory = torch.zeros((batch_size, self.cell_size), device=input_ids.device)
            full_batch_previous_state = torch.zeros((batch_size, self.hidden_size), device=input_ids.device)
        else:
            full_batch_previous_memory = initial_state[1] # (batch_size, cell_size)
            full_batch_previous_state = initial_state[0] # (batch_size, hidden_size)

        dropout_mask = self.dropout(full_batch_previous_state) if self.hidden_dropout_prob > 0.0 and self.training else None

        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else (total_timesteps - timestep - 1)

            previous_memory = full_batch_previous_memory.clone() # (batch_size, cell_size)
            previous_state = full_batch_previous_state.clone() # (batch_size, hidden_size)
            timestep_input = input_ids[:, index] # (batch_size, input_size)

            projected_input = self.input_linearity(timestep_input) # (batch_size, 4*cell_size)
            projected_state = self.state_linearity(previous_state) # (batch_size, 4*cell_size)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            input_gate = torch.sigmoid(projected_input[:, (0 * self.cell_size):(1 * self.cell_size)] +
                                       projected_state[:, (0 * self.cell_size):(1 * self.cell_size)]) # (batch_size, cell_size)
            forget_gate = torch.sigmoid(projected_input[:, (1 * self.cell_size):(2 * self.cell_size)] +
                                        projected_state[:, (1 * self.cell_size):(2 * self.cell_size)]) # (batch_size,cell_size)
            memory_init = torch.tanh(projected_input[:, (2 * self.cell_size):(3 * self.cell_size)] +
                                     projected_state[:, (2 * self.cell_size):(3 * self.cell_size)]) # (batch_size, cell_size)
            output_gate = torch.sigmoid(projected_input[:, (3 * self.cell_size):(4 * self.cell_size)] +
                                        projected_state[:, (3 * self.cell_size):(4 * self.cell_size)]) # (batch_size, cell_size)
            memory = input_gate * memory_init + forget_gate * previous_memory

            if self.memory_cell_clip_value:
                # pylint: disable=invalid-unary-operand-type
                memory = torch.clamp(memory, -self.memory_cell_clip_value, self.memory_cell_clip_value)

            pre_projection_timestep_output = output_gate * torch.tanh(memory) # (batch_size, cell_size)

            timestep_output = self.state_projection(pre_projection_timestep_output) # (batch_size, hidden_size)
            if self.state_projection_clip_value:
                # pylint: disable=invalid-unary-operand-type
                timestep_output = torch.clamp(timestep_output, -self.state_projection_clip_value, self.state_projection_clip_value)

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            #if dropout_mask is not None: timestep_output = timestep_output * dropout_mask

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            _index = torch.full((batch_size, ), index, device=input_ids.device)
            padding_mask = torch.gt(batch_lengths, _index).type(torch.FloatTensor).unsqueeze(1) # (batch_size)
            padding_mask = padding_mask.to(input_ids.device) if input_ids.device else padding_mask
            full_batch_previous_memory = memory*padding_mask + (1-padding_mask)*full_batch_previous_memory# (batch_size, cell_size)
            full_batch_previous_state = timestep_output*padding_mask + (1-padding_mask)*full_batch_previous_state # (batch_size, hidden_size)
            output_accumulator[:, index] = full_batch_previous_state # (batch_size, legnth, hidden_size)

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, ...). As this
        # LSTM cell cannot be stacked, the first dimension here is just 1.
        final_state = (full_batch_previous_state, full_batch_previous_memory) # (batch_size, length, hidden_size) & (batch_size, length, cell_size)

        return output_accumulator, final_state

class ElMoEncoder(nn.Module):
    def __init__(self, config):
        super(ElMoEncoder, self).__init__()
        forward_layer = LSTMCellWithProjection(config, True)
        backward_layer = LSTMCellWithProjection(config, False)
        self.forward_layers = nn.ModuleList([copy.deepcopy(forward_layer) for _ in range(config.num_hidden_layers)])
        self.backward_layers = nn.ModuleList([copy.deepcopy(backward_layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, batch_lengths, initial_states=None):
        if initial_states is None:
            initial_states = [None]*len(self.forward_layers)
        else:
            initial_states = list(zip(initial_states)) # need to change !!!

        forward_output_sequence = hidden_states
        backward_output_sequence = hidden_states
        final_states, sequence_outputs = [], []
        for layer_idx, state in enumerate(initial_states):
            forward_layer = self.forward_layers[layer_idx]
            backward_layer = self.backward_layers[layer_idx]
            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_initial_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state, backward_state = None, None

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence, batch_lengths, forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence, batch_lengths, backward_state)

            if layer_idx != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence, backward_output_sequence], -1)) # (batch_size, length, 2*hidden_size)

            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1))) # (batch_size, 2*hidden_size) & (batch_size, 2*cell_size)

        stacked_sequence_outputs = torch.stack(sequence_outputs) # (layer, batch_size, length, 2*hidden_size)
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple = (torch.stack(final_hidden_states, 0), torch.stack(final_memory_states, 0)) # ((layer, batch_size, 2*hidden_size), (layer, batch_size, 2*cell_size))
        return stacked_sequence_outputs, final_state_tuple
