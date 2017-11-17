import sys
from scipy.io import arff
import BayesLearner

_data_file = str(sys.argv[1])
_test_file = str(sys.argv[2])
_model = str(sys.argv[3])

data, meta = arff.loadarff(_data_file)
test_data, test_meta = arff.loadarff(_test_file)

if _model == 'n':
    myNaiveBayesLearner = BayesLearner.NaiveBayesLearner(meta)
    myNaiveBayesLearner.train(data)
    myNaiveBayesLearner.print_features()
    myNaiveBayesLearner.test(test_data)
elif _model == 't':
    myTANLearner = BayesLearner.TAN(meta)
    myTANLearner.train(data)
    myTANLearner.test(test_data)
else:
    print("Model type not correctly specified, please denote n or t.")
