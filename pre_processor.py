import numpy as np
import nltk

# To check if this package resource is present or not, if not then will throw error and download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Porter stemmer
from nltk.stem.porter import PorterStemmer

# split sentence into array of words/tokens
def tokenizer(input_sentence):  
    return nltk.word_tokenize(input_sentence)


# generates the root form of the word
def stem_it(word):
    l_word = word.lower()
    stemmer = PorterStemmer()
    return stemmer.stem(l_word)


def bag_of_words(tokenized_sentence, words):
    # creates a bog of words

    stemmed_words = [stem_it(word) for word in tokenized_sentence]
    # initialize a bag
    bag = np.zeros(len(words), dtype=np.float32)

    # populate present word into bag
    for index, w in enumerate(words):
        if w in stemmed_words: 
            bag[index] = 1

    return bag


# convert all words to stemmed words
def words_to_stem(all_words):
    words = []
    # remove special chars
    ignore_words = ['?', '.', '!']
    
    for w in all_words:
        if w not in ignore_words:
            words.append(stem_it(w))
    return words

def pre_processor_engine(intents):

    all_words = []
    tags = []
    xy = []

    # get sentence of patterns in intent
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(intent['tag'])
        for pattern in intent['patterns']:
            # tokenize sentence
            w = tokenizer(pattern)
            all_words.extend(w)
            # store associated words with its tag
            xy.append((w, tag))


    all_words = words_to_stem(all_words)


    # remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    #print(len(xy), "patterns")
    #print(len(tags), "tags:", tags)
    #print(len(all_words), "unique stemmed words:", all_words)

    # create training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        # X represents bag of words for each pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    # convert list into numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    processed_data = {}
    processed_data["all_words"] = all_words
    processed_data["tags"] = tags
    processed_data["xy"] = xy 
    processed_data["X_train"] = X_train
    processed_data["y_train"] = y_train

    return processed_data
