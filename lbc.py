from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import sys
import pprint

import pylab as pl

def main(argv):



   #squat v1 full
   cm = np.array([[0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0],
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
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#   cm = cm.transpose()
#   #print 'cm transpose()'
#   #pprint.pprint(cm)
   cm = normalize(cm)
#   # Show confusion matrix in a separate window
   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)
   plt.title('Squat v1 Full Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()


#   pl.matshow(cm)
#   pl.colorbar()
#   pl.title('Squat v1 Full Normalized Confusion Matrix')
#   pl.ylabel('True label')
#   pl.xlabel('Predicted label')
#   pl.show()

   #squat v1 simple
   cm = np.array([[121, 32], [86, 45]]) 
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window

   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)
   plt.title('Squat v1 Simple Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

   #bench v1 full
   cm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window
   pl.matshow(cm)
   pl.colorbar()
   pl.title('Bench v1 Full Normalized Confusion Matrix')
   pl.ylabel('True label')
   pl.xlabel('Predicted label')
   pl.show()

   #bench v1 simple
   cm = np.array([[49, 144], [135, 332]])
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window
   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)
   plt.title('Bench v1 Simple Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

   #press v1 full
   cm = np.array([[16, 10, 0, 0, 2, 0, 0, 13, 0, 7, 8, 0, 0, 0, 0],
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
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window
   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)
   plt.title('Press v1 Full Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

   #press v1 simple
   cm = np.array([[197, 87], [155, 116]])
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window

   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)
   plt.title('Press v1 Simple Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

   #squat v2 full
   cm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window
   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)

   plt.title('Squat v2 Full Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

   #squat v2 simple
   cm = np.array([[57, 32], [82, 212]])
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window
   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)

   plt.title('Squat v2 Simple Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

   #bench v2 full
   cm = np.array([[212, 0, 0, 0, 22, 19, 0, 26, 0, 1, 0, 0, 16, 0, 0, 0],
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
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window
   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)

   plt.title('Bench v2 Full Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

   #bench v2 simple
   cm = np.array([[1, 82], [22, 385]])
#   cm = cm.transpose()
   #print 'cm transpose()'
   #pprint.pprint(cm)
   cm = normalize(cm)
   # Show confusion matrix in a separate window
   fig = plt.figure()
   ax = fig.add_subplot(111)
   cax = ax.matshow(cm, interpolation='nearest', vmin = 0, vmax = 1)
   fig.colorbar(cax)

   plt.title('Bench v2 Simple Normalized Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

   #bench alt priorities full

   #bench alt priorities simple

def normalize(confusion_matrix):
   normal = confusion_matrix.tolist()
   for i, x in enumerate(confusion_matrix):
      row_total = 0
      for j, y in enumerate(x):
         row_total += y

      for j, y in enumerate(x):
         if row_total > 0:
            normal[i][j] /= (1.0 * row_total)

   pprint.pprint(normal)
   return normal

if __name__ == '__main__':
   main(sys.argv)
