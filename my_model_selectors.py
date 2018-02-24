import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        # Bayesian information criteria: BIC = −2 log L + p log N,
        # where L is the likelihood of the fitted model, p is the number of parameters,
        # and N is the number of data points. The term −2 log L decreases with
        # increasing model complexity (more parameters), whereas the penalties
        # p log N increase with increasing complexity. 

        # initialize lowest Bayesian Information Criterion value
        min_bic = float('inf')
        # initialize best model value
        best_model = None

        # select the best model for self.this_word based on BIC score for n between self.min_n_components and self.max_n_components
        for n in range(self.min_n_components, self.max_n_components + 1):
		    # Tip: The hmmlearn library may not be able to train or score all models. 
			# Implement try/except contructs as necessary to eliminate non-viable models from consideration.
            try:
                m = self.base_model(n)
                # L is the likelihood of the fitted model
                logL = m.score(self.X, self.lengths)
                # N is the number of data points: len(self.X)
                logN = np.log(len(self.X))
                # p is the number of parameters
				# source: https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/17
                # source: https://github.com/hmmlearn/hmmlearn/blob/master/hmmlearn/hmm.py
                p = n ** 2 + 2 * m.n_features * n - 1
                # Bayesian information criteria: BIC = −2 log L + p log N
                bic = -2 * logL + p * logN
                # Model selection: The lower the BIC value the better the model (only compare BIC with BIC values)
                if bic < min_bic:
                    min_bic, best_model  = bic, m               
            except:
                continue
        # return: GaussianHMM object
        # if exist a best model then return it
        if best_model:
            return best_model
        # else return base model
        else:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        # Discriminative Information Criterion (DIC) is an approximation of the DFC, where evidences
        # and anti-evidences are replaced by their BIC approximation. The DIC is the sum of two terms. 
        # The first term is a difference between the likelihood of the data and the average of anti-likelihood terms 
        # where the anti-likelihood of the data against model is a likelihood-like quantity 
        # in which the data and the model belong to competing categories.
        # The second term is zero when all data sets are of the same size. 
        # Considering that the first term contributes the most to the discriminative capabilities,
        # we used the following approximated version of the criterion: 
        # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i)) where
		# log(P(X(i)) is the difference between likelihood of the data
        # - 1/(M-1)SUM(log(P(X(all but i)) is the average of the anti-likelihood of the data
        # M is the likelihood component's length

        # initialize hightest Discriminative Information Criterion value
        max_dic = float('-inf')
        # initialize best model value
        best_model = None
        
		# for each component from min to max number of components
        for n in range(self.min_n_components, self.max_n_components + 1):
		    # Tip: The hmmlearn library may not be able to train or score all models. 
			# Implement try/except contructs as necessary to eliminate non-viable models from consideration.
            try:
				# get base model n
                m = self.base_model(n)
				# initialize scores list
                s = []
				# for each word in all_word_Xlengths
                for w, (X, lengths) in self.hwords.items():
				    # if the word is different from the word checked
                    if w != self.this_word:
					    # add score to the score list
                        s.append(m.score(X, lengths))
				# DIC is equal to the difference between likelihood of the data - the average of the anti-likelihood of the data
                dic = m.score(self.X, self.lengths) - np.mean(s)
				# if DIC is greater than previous highest DIC value than update max_dic and best_model values
                if dic > max_dic:
                    max_dic, best_model = dic, m
            # Tip: The hmmlearn library may not be able to train or score all models. 
			# Implement try/except contructs as necessary to eliminate non-viable models from consideration.
            except:
                continue
        # if there's a best model then return it
        if best_model:
            return best_model 
		# else return base model
        else:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
		
		# To estimate which topology model is better using only the training data, 
		# we can compare scores using cross-validation. 
		# One technique for cross-validation is to break the training set into "folds" 
		# and rotate which fold is left out of training. 
		# The "left out" fold scored. 
		# This gives us a proxy method of finding the best model to use on "unseen data".
		# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

		
		# inizialize highest cross-validation value
        max_cv = float('-inf')
		# inizialize best model value
        best_model = None
		
		# for each component from min to max number of components
        for n in range(self.min_n_components, self.max_n_components + 1):
		    # Tip: The hmmlearn library may not be able to train or score all models. 
			# Implement try/except contructs as necessary to eliminate non-viable models from consideration.
            try:			   
				# initialize scores list
                s = []
				# Tip: In order to run hmmlearn training using the X,lengths tuples on the new folds, 
		        # subsets must be combined based on the indices given for the folds. 
		        # A helper utility has been provided in the asl_utils module named combine_sequences for this purpose.
                for cv_train, cv_test in KFold(n_splits=2)(self.sequences):
                    # run hmmlearn training
                    self.X, self.lengths = combine_sequences(cv_train, self.sequences)
					# get actual base_model for n to train
                    m_train = self.base_model(n)
                    # run hmmlearn training					
                    X, lengths = combine_sequences(cv_test, self.sequences)
					# add train model score to the scores list
                    s.append(m_train.score(X, lengths))
                # calculate avarage scores for cross-validation
                cv = np.mean(s)
				# compare CV value with previous highest CV
                if cv > max_cv:
				    # if CV is greater than max_cv than update max_cv and best_model values
                    max_cv, best_model = cv, m_train
            # Tip: The hmmlearn library may not be able to train or score all models. 
			# Implement try/except contructs as necessary to eliminate non-viable models from consideration.
            except:
                continue
        # if there's a best model then return it
        if best_model:
            return best_model 
		# else return base model
        else:
            return self.base_model(self.n_constant)