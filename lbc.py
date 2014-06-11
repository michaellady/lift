from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import pylab as pl


cm = [[0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
             [6, 0, 0, 0, 0, 0, 0, 121, 0, 0, 15, 0, 11, 0],
              [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 9, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

print(cm)

# Show confusion matrix in a separate window
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()
