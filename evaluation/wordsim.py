import gzip
import sys

from ranking import *

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):
  word_vecs = {}
  if filename.endswith('.gz'): file_object = gzip.open(filename, 'r')
  else: file_object = open(filename, 'r')

  for line_num, line in enumerate(file_object):
    line = line.strip().lower()
    word = line.split()[0]
    word_vecs[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vec_val in enumerate(line.split()[1:]):
      word_vecs[word][index] = float(vec_val)
    ''' normalize weight vector '''
    word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return word_vecs


if __name__=='__main__':  
  word_vec_file = sys.argv[1]
  word_sim_file = sys.argv[2]
  
  print "Evaluating with : " + word_sim_file
  word_vecs = read_word_vectors(word_vec_file)
  print "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho"
  manual_dict, auto_dict = ({}, {})
  not_found, total_size = (0, 0)
  for line in open(word_sim_file,'r'):
    line = line.strip().lower()
    word1, word2, val = line.split()
    if word1 in word_vecs and word2 in word_vecs:
      manual_dict[(word1, word2)] = float(val)
      auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
    else:
      not_found += 1
    total_size += 1    
  print "%15s" % str(total_size), "%15s" % str(not_found),
  print "%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))
  print ''
