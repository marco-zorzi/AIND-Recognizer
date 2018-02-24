import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set
   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
	
	# for each X and lengths in all word lengths values
    for X, lengths in test_set.get_all_Xlengths().values():
	    # initialize word Liklihood dictionary
        wl = {}
		# initialize highest score value
        max_s = float('-inf')
		# initialize guess 
        g = None
		# for each word and model in models items
        for w, m in models.items():
            try:
			    # get score from model score
                s = m.score(X, lengths)
				# word Liklihood for this word is equel to the score
                wl[w] = s
				# compare score to previus hightest score
                if s > max_s:
				    # if score is greater than previus score than update max_s value and guess word
                    max_s, g  = s, w                     
            except:
                wl[w] = float('-inf')
		# add guess word to guesses list
        guesses.append(g)
		# add word Liklihood to probabilities list
        probabilities.append(wl)
    # return probabilities, guesses
    return probabilities, guesses
    