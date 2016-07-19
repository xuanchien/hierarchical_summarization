import os
import gensim
from gensim import corpora
from nltk.stem.porter import *

stemmer = PorterStemmer()

directory = "research/data/summarization/dailymail/training"
files = os.listdir(directory)

dictionary = corpora.Dictionary()

for file in files:
	print "processing file: ", file
	if file.startswith("."):
		continue

	try:
		tokens = open(os.path.join(directory, file), "r").read().decode('utf-8').lower().split()
		tokens = [stemmer.stem(token) for token in tokens]
		dictionary.add_documents([tokens], prune_at=None) #keep VOCAB SIZE <= 10000
	except UnicodeDecodeError:
		print "Error while processing file ", file, ". Ignoring it"

dictionary.filter_extremes(no_below=3)
dictionary.save("vocab.dict")