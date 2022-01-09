

import codecs
import csv
import re
import sys
import pickle
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# Load tokenizer
import pickle
def remove_tone_line(s):
  # Trong tiếng Việt chỉ thêm dấu vào nguyên âm và y, d
  intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
  intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
  intab = list(intab_l+intab_u)

  # Thay thế các ký tự ở trên lần lượt bằng các ký tự bên dưới
  outtab_l = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d" 
  outtab_u = "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"
  outtab = outtab_l + outtab_u

  r = re.compile("|".join(intab)) # khớp với một trong các ký tự của intab
  replace_dict = dict(zip(intab, outtab)) # Dictionary có key-value là từ có dấu-từ không dấu. VD: {'â' : 'a'}
  # Thay thế các từ có dấu xuất hiện trong tìm kiếm của regex bằng từ không dấu tương ứng
  return r.sub(lambda m: replace_dict[m.group()], s)
  # m là kết quả của so khớp, m.group trả về các phần tử khớp với pattern

import pickle

def _save_pickle(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pickle(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj



# thêm token start và end vào câu input và câu target, index của start và end là 2 chỉ số cuối của từ điển
def encode(ipt, opt):
  ipt = [tokenizer_ipt.vocab_size] + tokenizer_ipt.encode(ipt.numpy()) + [tokenizer_ipt.vocab_size+1]

  opt = [tokenizer_opt.vocab_size] + tokenizer_opt.encode(opt.numpy()) + [tokenizer_opt.vocab_size+1]
  
  return ipt, opt

def tf_encode(ipt, opt):
  result_ipt, result_opt = tf.py_function(encode, [ipt, opt], [tf.int64, tf.int64])
  result_ipt.set_shape([None])
  result_opt.set_shape([None])
  return result_ipt, result_opt

# Loại bỏ những mẫu có kích thước lớn hơn 40 tokens để quá trình huấn luyện nhanh hơn
MAX_LENGTH = 40

# pos là chỉ số của từ hiện tại 
# i là chỉ số của phần tử trong vector encoding (một từ được encode thành 1 vector)
# dmodel - kích thước của vector encoding

def get_angles(pos, i, dmodel):
  # tính phần bên trong hàm sin or cos
  angle_rates = 1/np.power(10000, (2*(i//2))/np.float32(dmodel))
  return pos * angle_rates

def positional_encoding(position, dmodel):
  # position là số từ cần tính
  # tạo ra ma trận mỗi hàng biểu diễn positional encoding vector của một từ
  angle_rates = get_angles(np.arange(position)[:, np.newaxis], # số hàng bằng số từ
                           np.arange(dmodel)[np.newaxis, :], # số cột bằng len(encoding vector)
                           dmodel) # shape (position, dmodel)
  # apply sin to even indices: 2i
  angle_rates[:, 0::2] = np.sin(angle_rates[:, 0::2])
  # appy cos to odd indices: 2i+1
  angle_rates[:, 1::2] = np.cos(angle_rates[:, 1::2])

  pos_encoding = angle_rates[np.newaxis, ...] 
  return tf.cast(pos_encoding, dtype=tf.float32) # shape(1, position, dmodel)

# Tạo mask các từ được padding vào câu
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# Tạo mask các từ tương lai
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True) # ma trận k được transpose trước khi nhân (..., seq_len_q, seq_len_k), 
                                                # chỉ nhân tương ứng hai chiều cuối cùng

  # Scale matmul
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor
  if mask is not None:
    scaled_attention_logits += mask * (-1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_2)
  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    # d_model là chiều dài của vector mã hóa mỗi từ
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    # vector x có chiều (batch_size, seq_len, d_model) --> chia chiều dmodel thành num_head x depth
    # Split the last dimension into (num_heads, depth)
    # Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    # v, k, q có dim=(batch_size, seq_len, d_model)
    batch_size = tf.shape(q)[0]
    
    # Dense áp dụng lên chiều cuối cùng của mảng truyền vào

    q = self.wq(q) # (batch_size, seq_len, d_model)
    k = self.wk(k) # (batch_size, seq_len, d_model)
    v = self.wv(v) # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    # output của Multi-head attention
    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights
    # Như vậy shape của attention sẽ là (batch_size, num_head, seq_len_q, seq_len_k) 
    # và shape của output sẽ là (batch_size, seq_len_q, d_model)

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    attn_output, _ = self.mha(x, x, x, mask) # (batch_size, seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output) # (batch_size, seq_len, d_model)

    ffn_output = self.ffn(out1) # (batch_size, seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output) 

    return out2 # (batch_size, seq_len, d_model)

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
  
  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    # enc_output.shape = (batch_size, input_seq_len, d_model)
    attn1, attn_weight_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weight_block2 = self.mha2(enc_output, enc_output, out1, padding_mask) # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1) # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2) # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training = training)
    out3 = self.layernorm3(ffn_output + out2) # (batch_size, target_seq_len, d_model)

    return out3, attn_weight_block1, attn_weight_block2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff,
               input_vocab_size, maximum_position_encoding, rate=0.1):
    # num_layers là số encoder layer xếp chồng lên nhau
    # input_vocab_size: số từ trong từ điển
    # maximum_position_encoding: số từ cần tính position encoding
    # dff: số node trong layer đầu tiên của feed forward

    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model) # shape(1, position/seq_len, dmodel)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                      for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    # x (batch_size, input_seq_len, d_model)
    # mask
    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding
    x = self.embedding(x) # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x #(batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads,
               dff, target_vocab_size, maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x) # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
              target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                            input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                            target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    # print('enc_padding_mask: ', enc_padding_mask)
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_ipt.vocab_size + 2
target_vocab_size = tokenizer_opt.vocab_size + 2
dropout_rate = 0.1

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = step ** -0.5
    arg2 = step * self.warmup_steps ** -1.5

    return self.d_model ** -0.5 * tf.math.minimum(arg1, arg2)



def create_masks(inp, tar):
  # vd: inp(64, 37), tar(64, 38)

  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp) # (64, 1, 1, 37)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp) # (64, 1, 1, 37)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1]) # (38, 38)
  dec_target_padding_mask = create_padding_mask(tar) # (64, 1, 1, 38)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask) # (64, 1, 38, 38) -> cả những từ tương lai và cả những từ được pad 0 có mask bằng 1
  return enc_padding_mask, combined_mask, dec_padding_mask

# def _save_pickle(path, obj):
#   with open(path, 'wb') as f:
#     pickle.dump(obj, f)

# def _load_pickle(path):
#   with open(path, 'rb') as f:
#     obj = pickle.load(f)
#   return obj

# tokenizer_ipt = _load_pickle('tokenizer/tokenizer_ipt.pkl')
# tokenizer_opt = _load_pickle('tokenizer/tokenizer_opt.pkl')

# Khai báo tham số
num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_ipt.vocab_size + 2
target_vocab_size = tokenizer_opt.vocab_size + 2
dropout_rate = 0.1
learning_rate = 0.01

# Load model
transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                          input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

checkpoint_path = "./checkpoints/train_500k_version3"

ckpt = tf.train.Checkpoint(transformer=transformer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)

def evaluate(inp_sentence):
  start_token = [tokenizer_ipt.vocab_size]
  end_token = [tokenizer_ipt.vocab_size+1]
  
  # inp sentence is non_diacritic, hence adding the start and end token
  inp_sentence = start_token + tokenizer_ipt.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # as the target is exist diacritic, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_opt.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_opt.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


def preprocessing(str_input):
  
  intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
  intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
  intab = list(intab_l+intab_u)

  outtab_l = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d" 
  outtab_u = "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"
  outtab = outtab_l + outtab_u

  # remove whitespace
  str_input = ' '.join(str_input.split())

  str_restore = '''!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ''' + intab_l + intab_u
  d_restore = dict()

  for i, v in enumerate(str_input):
    if v in str_restore:
      d_restore[i] = v

  # xóa dấu nếu có
  r = re.compile("|".join(intab))
  replace_dict = dict(zip(intab, outtab))
  str_input = r.sub(lambda m: replace_dict[m.group()], str_input)

  # xóa các ký tự đặc biệt
  for v in d_restore.values():
    if v != ' ':
      str_input = str_input.replace(v, '')

  return str_input.strip(), d_restore


def postprocessing(str_output, d_restore, restore_tone=False, keep_special_character=True):
  str_restore = '''!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '''

  index_special_character = []

  str_output = str_output.replace(' ', '')
  d_restore = OrderedDict(sorted(d_restore.items()))

  for k, v in d_restore.items():
    if v in str_restore:
      str_output = str_output[:k] + v + str_output[k:]
      if v != ' ':
        index_special_character.append(k)
    if restore_tone:
      str_output = str_output.replace(str_output[k], v)

  str_output = np.array(list(str_output))
  if not keep_special_character:
    str_output[index_special_character] = ''

  return ''.join(str_output)


def add_diacritic(sentence, restore_tone, keep_special_character):
  # preprocessing
  sentence, d_restore = preprocessing(sentence)

  sentence = ' '.join(sentence.split())
  result, attention_weights = evaluate(str(sentence))
  predicted_sentence = tokenizer_opt.decode([i for i in result 
                                            if i < tokenizer_opt.vocab_size])  
  predicted_sentence = predicted_sentence.replace('.', '').replace(',', '')
  # postprocessing
  result = postprocessing(predicted_sentence, d_restore, restore_tone, keep_special_character)

  # print('Input: {}'.format(sentence))
  # print('Predicted translation: {}'.format(predicted_sentence))
  
  return result


def split(s):
  list_paragraph = []
  list_split = []
  start = 0
  for i, v in enumerate(s):
    if v in [',', '.'] and not (s[i-1].isdigit() and s[i+1].isdigit()):
      list_split.append(v)
      list_paragraph.append(s[start:i])
      start = i+1

  if s[start:] != '':
    list_paragraph.append(s[start:])
  return list_paragraph, list_split


def predict(s, restore_tone=False, keep_special_character=True):
  list_paragraph, list_split = split(s)
  result = ''
  for i, s in enumerate(list_paragraph):
    if s == '':
      result = result.strip()
    if s != '':
      result += add_diacritic(s, restore_tone, keep_special_character).replace('\n', '')
    if len(list_split) != 0:
      result += list_split.pop(0)
      result += ' '

  return result


if __name__ == '__main__':

  result = predict("thoi tiet hom nay thich hop de di choi da ngoai")
  print(result)









