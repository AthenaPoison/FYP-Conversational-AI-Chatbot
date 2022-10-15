# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
from io import open
import itertools
import math 
import nltk
from nltk.translate.bleu_score import corpus_bleu
from flask import Flask, render_template, request, jsonify
import json
# %%
#gpu check

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

# %%
#from google.colab import drive
#drive.mount('/drive')

corpus_name = "covid"
corpus = os.path.join(corpus_name)
def printLines(file, n=20):
  with open(file, 'rb') as datafile:
    lines = datafile.readlines()
  for line in lines[:n]:
    print(line)
data_file = os.path.join(corpus, "Covid-Dataset_full.txt")

# %%
#mapping unique words
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
  def __init__(self, name):
    self.name = name
    self.trimmed = False
    self.word2index = {}
    self.word2count = {}
    self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
    self.num_words = 3 #counting SOS, EOS, PAD

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.num_words
      self.word2count[word] = 1
      self.index2word[self.num_words] = word
      self.num_words += 1
    else: 
      self.word2count[word] += 1
  
  #remove words below a certain threshold
  def trim(self, min_count):
    if self.trimmed:
      return
    self.trimmed = True

    keep_words = []

    for k, v in self.word2count.items():
      if v >= min_count:
        keep_words.append(k)
    
    #reinitialize dictionaries
    self.word2index = {}
    self.word2count = {}
    self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
    self.num_words = 3 #count default tokens

    for word in keep_words:
      self.addWord(word)

# %%
MAX_LENGTH = 30 # max sentence length to consider 

#turn a unicode string to plain ASCII 
def unicodeToAscii(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
  )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

#read query/response pairs and return voc object
def readVocs(datafile, corpus_name):
  print("Reading lines...")
  #read the file and split the lines
  lines = open(datafile, encoding='utf-8').\
    read().strip().split('\n')
  #split every line into pairs and normalize
  pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
  voc = Voc(corpus_name)
  return voc, pairs

#return true if both sentence in pair 'p' are under the max_length threshold
def filterPair(p):
  #input sequences need to preserve the last word for EOS token
  return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

#filter pairs using filterpair condition
def filterPairs(pairs):
  return [pair for pair in pairs if filterPair(pair)]

#USing the funtions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
  voc, pairs = readVocs(datafile, corpus_name)
  pairs = filterPairs(pairs)
  for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
  return voc, pairs

#load/assemble voc and pairs
save_dir = os.path.join("/drive/My Drive", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, data_file, save_dir)

# %%
MIN_COUNT = 1

def trimRareWords(voc, pairs, MIN_COUNT):
  #trim words used under the min_count from the voc
  voc.trim(MIN_COUNT)
  #filter out pairs with trimmed words
  keep_pairs = []
  for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True
    #check input sentence
    for word in input_sentence.split(' '):
      if word not in voc.word2index:
        keep_input = False
        break

    #only keep pairs that do not contain trimmed words in their input or output sentence
    if keep_input and keep_output:
      keep_pairs.append(pair)

  return keep_pairs

#trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

# %%
#convert words to index for tensor 
def indexesFromSentence(voc,sentence): 
  wordIndex = []
  for word in sentence.split(' '):
    if word in voc.word2index:
      wordIndex.append(voc.word2index[word])
  return wordIndex + [EOS_token]

#pad empty space in batch with zeros after eos
def zeroPadding(l, fillvalue=PAD_token):
  return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
  m = []
  for i, seq in enumerate(l):
    m.append([])
    for token in seq:
      if token == PAD_token:
        m[i].append(0)
      else:
        m[i].append(1)
  return m

#returns padded input sequence of tensor and lengths
def inputVar(l, voc):
  indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
  lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
  padList = zeroPadding(indexes_batch)
  padVar = torch.LongTensor(padList)
  return padVar, lengths

#returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
  indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
  max_target_len = max([len(indexes) for indexes in indexes_batch])
  padList = zeroPadding(indexes_batch)
  mask = binaryMatrix(padList)
  mask = torch.BoolTensor(mask)
  padVar = torch.LongTensor(padList)
  return padVar, mask, max_target_len

#returns all items for a given batch of pairs

def batch2TrainData(voc, pair_batch):
  pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
  input_batch, output_batch = [], []
  for pair in pair_batch:
    input_batch.append(pair[0])
    output_batch.append(pair[1])
  inp, lengths = inputVar(input_batch, voc)
  output, mask, max_target_len = outputVar(output_batch, voc)
  return inp, lengths, output, mask, max_target_len

#example
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

# %%
class EncoderRNN(nn.Module):
  def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
    super(EncoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.embedding = embedding

    #initialize gru; input_size and hidden_size params are both set to hidde_side
    #because our input size is a word embedding with number of features == hidden_size
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                      dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
    
  def forward(self, input_seq, input_lengths, hidden=None):
    #convert word indexes to embddings
    embedded = self.embedding(input_seq)
    #pack padded batch of sequences for RNN module
    packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
    # forward pass through gru
    outputs, hidden = self.gru(packed, hidden)
    #unpacking padding
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
    #sum bidirectional gru outputs
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
    #return output and final hidden state
    return outputs, hidden

# %%
class Attn(nn.Module):
  def __init__(self, method, hidden_size):
    super(Attn, self).__init__()
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, "is not an appropriate attention method.")
    self.hidden_size = hidden_size
    if self.method == 'general':
      self.attn = nn.Linear(self.hidden_size, hidden_size)
    elif self.method == 'concat':
      self.attn = nn.Linear(self.hidden_size*2, hidden_size)
      self.v = nn.Parameter(torch.FloatTensor(hidden_size))
  
  def dot_score(self, hidden, encoder_output):
    return torch.sum(hidden*encoder_output, dim=2)

  def general_score(self, hidden, encoder_output):
    energy = self.attn(encoder_output)
    return torch.sum(hidden*energy, dim=2)

  def concat_score(self, hidden, encoder_output):
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
                                  encoder_output), 2)).tanh()
    return torch.sum(self.v*energy, dim=2)

  def forward(self, hidden, encoder_outputs):
    #calculate the attention weights(energies) based on  the given method
    if self.method == 'general':
      attn_energies = self.general_score(hidden, encoder_outputs)
    elif self.method == 'concat':
      attn_energies = self.concat_score(hidden, encoder_outputs)
    elif self.method == 'dot':
      attn_energies = self.dot_score(hidden, encoder_outputs)

    #transpose max_length and batch_size dimensions
    attn_energies = attn_energies.t()

    #return the softmx normalized probability scores(with added dimensions)
    return F.softmax(attn_energies, dim=1).unsqueeze(1)

# %%
class LuongAttnDecoderRNN(nn.Module):
  def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
    super(LuongAttnDecoderRNN, self).__init__()

    #keep for reference
    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout
    
    #define layers
    self.embedding = embedding
    self.embedding_dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
    self.concat = nn.Linear(hidden_size*2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

    self.attn = Attn(attn_model, hidden_size)

  def forward(self, input_step, last_hidden, encoder_outputs):
    #one word at a time
    #get embedding of current input word
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    #forward through unidirectional gru
    rnn_output, hidden = self.gru(embedded, last_hidden)
    #calculate attention weights from the current gru output
    attn_weights = self.attn(rnn_output, encoder_outputs)
    #multiply attention weights to encoder outputs to get new "weighted sum" context vector
    context = attn_weights.bmm(encoder_outputs.transpose(0,1))
    #concatenate weighted context vector and gru output using luong eq.5
    rnn_output = rnn_output.squeeze(0)
    context = context.squeeze(1)
    concat_input = torch.cat((rnn_output, context), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    #predict next word using luong eq.6
    output = self.out(concat_output)
    output = F.softmax(output, dim=1)
    #return output and final hidden state
    return output, hidden

# %%
def maskNLLLoss(inp, target, mask):
  nTotal = mask.sum()
  crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
  loss = crossEntropy.masked_select(mask).mean()
  loss = loss.to(device)
  return loss, nTotal.item()

# %%
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
  
  #zero gradient
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  #set device options
  input_variable = input_variable.to(device)
  target_variable = target_variable.to(device)
  mask = mask.to(device)
  # lengths for rnn packing should always be on the cpu
  lengths = lengths.to("cpu")

  #initialize variables
  loss = 0
  print_losses = []
  n_totals = 0

  #forward pass through the encoder
  encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

  #create initial decoder input(start with SOS tokens for each sentence)
  decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
  decoder_input = decoder_input.to(device)

  #set initia decoder hidden state to the encoder's final hidden state
  decoder_hidden = encoder_hidden[:decoder.n_layers]

  #determine if we are using teacher forcing this iteration
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  #forward batch sequence through decoder one time step at a time
  if use_teacher_forcing:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
      #teacher forching: next input is current target
      decoder_input = target_variable[t].view(1, -1)
      #calculate and accumulate loss
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_loss
      print_losses.append(mask_loss.item()*nTotal)
      n_totals += nTotal
  else: 
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
      #no teacher forcing: next input is decoder's own current output
      _, topi = decoder_output.topk(1)
      decoder_input = torch.LongTensor([[topi[1][0] for i in range(batch_size)]])
      decoder_input = decoder_input.to(device)
      #calculate and accumulate loss
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_loss
      print_losses.append(mask_loss.item()*nTotal)
      n_totals += nTotal
  
  #perform backpropagation
  loss.backward()

  #clip gradients: gradients are modified in place
  _= nn.utils.clip_grad_norm(encoder.parameters(), clip)
  _= nn.utils.clip_grad_norm(decoder.parameters(), clip)

  #adjust model weights
  encoder_optimizer.step()
  decoder_optimizer.step()

  return sum(print_losses)/n_totals

# %%
losses = []
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, 
               save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):
  
  #load batches for each iteration
  training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

  #initializing
  print('Initializing...')
  start_iteration = 1
  print_loss = 0
  if loadFilename:
    start_iteration = checkpoint['iteration']+1
  
  #training loop
  print('Training...')
  for iteration in range(start_iteration, n_iteration+1):
    training_batch = training_batches[iteration-1]
    #extract fields from batch
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    #run a training iteration with batch
    loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                 decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
    print_loss += loss

    #print progress
    if iteration % print_every == 0:
      print_loss_avg = print_loss/ print_every
      print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
        iteration/n_iteration*100, print_loss_avg))
      losses.append(print_loss_avg)
      print_loss = 0

      #save checkpoint
    if (iteration % save_every == 0):
        directory = os.path.join(save_dir, model_name, corpus_name,
              '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
        if not os.path.exists(directory):
          os.makedirs(directory)
        torch.save({
            'iteration': iteration,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc.__dict__,
            'embedding': embedding.state_dict(),
            'losses': losses
        }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

# %%
class GreedySearchDecoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(GreedySearchDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input_seq, input_length, max_length):
    #forward input through encoder model
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
    #prepare encoder's final hidden layer to be first hidden input to th decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    #initialize decoder input with SOS_token
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long)*SOS_token
    #initialize tensors to append decoded words to
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    #iteratively decode one word token at a time
    for _ in range(max_length):
      #forward pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
      #obtain most likely word token and its softmax score
      decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
      #record token and score
      all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
      all_scores = torch.cat((all_scores, decoder_scores), dim=0)
      #prepare current token to be next decoder input (add a dimension)
      decoder_input = torch.unsqueeze(decoder_input, 0)
    
    #return collections of word tokens and scores
    return all_tokens, all_scores

# %%
score =  []
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
  ###format input sentence as batch
  indexes_batch = [indexesFromSentence(voc, sentence)]
  #Create lengths tensor
  lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
  #transpose dimensions of batch to match models' expectations
  input_batch = torch.LongTensor(indexes_batch).transpose(0,1)
  #use appropriate device
  input_batch = input_batch.to(device)
  lengths = lengths.to("cpu")
  #decode sentence with searcher
  tokens, scores = searcher(input_batch, lengths, max_length)
  #indexes -> words
  decode_words = [voc.index2word[token.item()] for token in tokens]
  return decode_words
response_words = []

def evaluateInput(encoder, decoder, searcher, voc, input_sentence):
  while(1):
    try:
      #check if it is quit case
      if input_sentence == 'q' or input_sentence == 'quit': break
      #normalize sentence
      input_sentence = normalizeString(input_sentence)
      #evaluate sentence
      output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
      #format and print response sentence
      output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
      reference = []
      for pair in pairs:
        reference.append(pair[1].split( ))
      score.append(corpus_bleu([reference], [output_words]))
      reply_words = ' '.join(output_words)
      return reply_words
    except KeyError:
      print("Error: Encountered unknown word.")

# %%
#configure models
model_name = 'cb_model_full'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

#set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 10000
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                          '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                          '{}_checkpoint.tar'.format(checkpoint_iter))

loadFilename = os.path.join('{}_checkpoint.tar'.format(checkpoint_iter))
#load model if loadFilename is provided

if loadFilename:
  checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
  #if loading on same machine the model was trained on
  #checkpoint = torch.load(loadFilename)
  #if loading a model trained on GPU to CPU
  
  encoder_sd = checkpoint['en']
  decoder_sd = checkpoint['de']
  encoder_optimizer_sd = checkpoint['en_opt']
  decoder_optimizer_sd = checkpoint['de_opt']
  embedding_sd = checkpoint['embedding']
  voc.__dict__ = checkpoint['voc_dict']
 # losses.append(checkpoint['losses'])


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# %%
# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 2000
print_every = 1
save_every = 500

#ensure dropout layers are in train mode
encoder.train()
decoder.train()

#initialize optimizers
print('Building optimizers...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate*decoder_learning_ratio)
if loadFilename:
  encoder_optimizer.load_state_dict(encoder_optimizer_sd)
  decoder_optimizer.load_state_dict(decoder_optimizer_sd)

#run training iterations
if not loadFilename:
  print("Starting training!")
  trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)
  
else:
  print("Training loaded")
  

# %%
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chatbot.html')

@app.route('/getprediction',methods=['POST', 'GET'])
def getprediction():    
    input = request.get_json()
    input = json.loads(input)
    reply = evaluateInput(encoder, decoder, searcher, voc, input)
    return jsonify(reply)



