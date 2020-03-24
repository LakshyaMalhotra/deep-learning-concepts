# utils.py
import re
from collections import Counter

def preprocess(text):
    """
    Replace punctuation with tokens so we can use them in our model
    """
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace(':', ' <COLON> ')

    words = text.split()

    # Remove all the words with 5 or fewer occurences
    word_counts = Counter(words)
    # trimmed_words = [words for words, freq in word_counts.items() if freq > 5]
    trimmed_words = [word for word in words if word_counts[word] > 5]
    
    return trimmed_words

def create_look_up_tables(words):
    """
    Create look-up tables for the vocabulary
    :param words: Input list of words
    :return: A tuple of dicts
    """
    word_counts = Counter(words)
    # sort the words from most to least frequent in text occurence
    sorted_words = sorted(word_counts, key=lambda x: word_counts[x], reverse=True)
    # create int2vocab dictionaries
    int2vocab = dict(enumerate(sorted_words))
    vocab2int = {ch: ii for ii, ch in int2vocab.items()}

    return int2vocab, vocab2int
