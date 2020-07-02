#!/bin/sh

# eduskunta-vkk dataset
# https://github.com/aajanki/eduskunta-vkk
mkdir -p data/vkk
wget -P data/vkk https://github.com/aajanki/eduskunta-vkk/raw/v2/vkk/train.csv.bz2
wget -P data/vkk https://github.com/aajanki/eduskunta-vkk/raw/v2/vkk/dev.csv.bz2
wget -P data/vkk https://github.com/aajanki/eduskunta-vkk/raw/v2/vkk/test.csv.bz2

# Finnish Internet Parsebank word2vec vectors
# http://bionlp.utu.fi/finnish-internet-parsebank.html
mkdir -p data/fin-word2vec/
wget -P data/fin-word2vec/ http://bionlp-www.utu.fi/fin-vector-space-models/fin-word2vec.bin

# Word frequencies on Finnish Internet Parseback
# https://turkunlp.org/finnish_nlp.html
mkdir -p data/finnish_vocab
wget -P data/finnish_vocab/ http://bionlp-www.utu.fi/.jmnybl/finnish_vocab.txt.gz
