from nrclex import NRCLex

#Instantiate text object (for best results, 'text' should be unicode).

text_object = NRCLex('I want good food!')

#Return words list.

text_object.words
text_object.raw_emotion_scores

# Problem: working with just a sentence in jupyter notebook but not sure how to use it on a dataset !!