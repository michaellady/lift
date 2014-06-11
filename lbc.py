from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import pylab as pl

#squat v1 full
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
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Squat v1 Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#squat v1 simple
cm = [[121, 32], [86, 45]] 
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Squat v1 Simple Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#bench v1 full
cm = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 20, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 212, 115, 0, 9, 0, 29, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 122, 49, 0, 22, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 38, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Bench v1 Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#bench v1 simple
cm = [[49, 144], [135, 332]]
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Bench v1 Simple Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#press v1 full
cm = [[16, 10, 0, 0, 2, 0, 0, 13, 0, 7, 8, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [3, 1, 0, 0, 17, 0, 0, 47, 0, 5, 0, 0, 0, 3, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [10, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0],
             [14, 5, 0, 0, 38, 0, 17, 197, 0, 4, 5, 0, 0, 4, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 3, 0, 0, 33, 0, 2, 0, 0, 0, 0, 0],
                [7, 10, 0, 0, 2, 0, 0, 32, 0, 0, 10, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Press v1 Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#press v1 simple
cm = [[197, 87], [155, 116]]
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Press v1 Simple Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#squat v2 full
cm = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 57, 2, 0, 30, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 13, 0, 0, 0, 62, 0, 0, 95, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 8, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Squat v2 Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#squat v2 simple
cm = [[57, 32], [82, 212]]
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Squat v2 Simple Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#bench v2 full
cm = [[212, 0, 0, 0, 22, 19, 0, 26, 0, 1, 0, 0, 16, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [13, 0, 0, 7, 9, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
           [73, 0, 0, 9, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [7, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Bench v2 Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#bench v2 simple
cm = [[1, 82], [22, 385]]
# Show confusion matrix in a separate window
pl.matshow(cm)
pl.colorbar()
pl.title('Bench v2 Simple Confusion matrix')
pl.ylabel('Predicted label')
pl.xlabel('True label')
pl.show()

#bench alt priorities full

#bench alt priorities simple


