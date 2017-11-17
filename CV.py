from scipy.io import arff
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import BayesLearner
import matplotlib.pyplot as plt
from scipy.interpolate import spline

data, meta = arff.loadarff('chess-KingRookVKingPawn.arff')

_folds = 10

# Set up figure.
fig = plt.figure()

fig.suptitle('Part 2: CV Accuracies', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Accuracies')
ax.set_xlabel('held out set label')
ax.set_ylabel('accuracy')

np.random.shuffle(data)

size = len(data)
data_sets = dict()
train_sets = dict()
for i in range(_folds):
    data_sets[i] = []
    train_sets[i] = []

i = 0
for x in data:
    data_sets[int((_folds * i)/size)].append(x)
    i += 1

for i in range(_folds):
    data_sets[i] = np.array(data_sets[i])

for i in range(_folds):
    for j in range(_folds):
        if i != j:
            for x in data_sets[j]:
                train_sets[i].append(x)
    train_sets[i] = np.array(train_sets[i])


_accuracies_NB = []
_accuracies_TAN = []

#for i in range(_folds):
    # Give neural network the learning rate.
    # NOTE: Data must have a target feature 'Class' in the last indice.
    #myNB.train(train_sets[i])
    #_accuracies_NB.append(myNB.test(data_sets[i]))

    #myTAN = BayesLearner.TAN(meta)
    #myTAN.train(train_sets[i])
    #_accuracies_TAN.append(myTAN.test(data_sets[i]))

#print(_accuracies_NB)
#print(_accuracies_TAN)
_accuracies_NB = [0.890625, 0.896875, 0.8432601880877743, 0.88125, 0.890282131661442, 0.88125, 0.86875, 0.8808777429467085, 0.884375, 0.8808777429467085]
_accuracies_TAN = [0.946875, 0.940625, 0.9122257053291536, 0.928125, 0.9059561128526645, 0.915625, 0.915625, 0.9278996865203761, 0.93125, 0.9404388714733543]
ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], _accuracies_NB, 'o', label="Naive Bayes")
ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], _accuracies_TAN, 'x', label="TAN")
ax.legend()
ax.axis([0, 10, 0, 1])

# Plot.
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
