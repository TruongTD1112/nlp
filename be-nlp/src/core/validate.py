import codecs
import csv
import re
import sys
import string
import re

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

# https://realpython.com/python-encodings-guide/
# List các ký tự hợp lệ trong tiếng Việt
intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'
punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
whitespace = ' '
accept_strings =  intab_l + ascii_lowercase + digits + punctuation + whitespace
r = re.compile('^[' + accept_strings + ']+$')


# Một câu sẽ được coi là hợp lệ nếu có các ký tự nằm trong accept_strings
def _check_tieng_viet(seq):
  if re.match(r, seq.lower()):
    return True
  else:
    return False


def preprocessing(string):
  listStringSplit = string.split(" ")
  indexInit = []
  for i in range(len(listStringSplit)):
    indexInit.append(i)

  init = list(zip(listStringSplit, indexInit))

  listStringAfterRemove = []
  listStringRemove = []
  for index, value in enumerate(init):
    if(_check_tieng_viet(value[0])):
      listStringAfterRemove.append((value))
    else:
      listStringRemove.append(value)
  # print(listStringAfterRemove)
  # print(listStringRemove)
  stringAfterRemove = ""
  for index, value in enumerate(listStringAfterRemove):
    stringAfterRemove += value[0] + " "

  return [stringAfterRemove, listStringAfterRemove, listStringRemove, len(listStringSplit)]


# def getResult(string):
#   preProcess = preprocessing(string)
#   stringPredict = add_diacritic()
#   stringPreProcess = preProcess[0]





# print(init)
# print(_check_tieng_viet('tiếng'))