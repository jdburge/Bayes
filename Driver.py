from scipy.io import arff
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import BayesLearner

data, meta = arff.loadarff('lymph_train.arff')
test_data, test_meta = arff.loadarff('lymph_test.arff')

#myNaiveBayesLearner = BayesLearner.NaiveBayesLearner(meta)
#myNaiveBayesLearner.train(data)
#myNaiveBayesLearner.print_features()
#myNaiveBayesLearner.test(test_data)

myTANLearner = BayesLearner.TAN(meta)
myTANLearner.train(data)
myTANLearner.test(test_data)

