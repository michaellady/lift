import sys
import pickle
import sqlite3
import pprint
import pylab as pl
import scipy as sp
from scipy import stats
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import argparse

exercises = {'press': 0, 'bench' : 1, 'squat' : 2}

def main(argv):
   parser = argparse.ArgumentParser(description='Set up how LIFT runs')
   parser.add_argument('exercise', 
         help='enter exercise to analyze (press, bench, squat)')
   parser.add_argument('database_file', help='enter database filename to use')
   parser.add_argument('-o','--one', help='run leave one subject out validation (default: cross validation)', action="store_true")

   args = parser.parse_args()
   cur, conn = setup_db_cursor(args.database_file)
   moments = get_moments(cur, conn, exercises[args.exercise])
   print 'moments'
#   pprint.pprint(moments)
   print 'len moments: '+str(len(moments))

   if args.one:
      print 'leave one out vaidation'
      athletes = get_athletes(cur, conn)
      result_accuracy = leave_one_out(athletes, moments)
      print 'result_accuracy '+str(result_accuracy)

   else:
      data_target_list = get_data_target_list(moments)
      data_target_list = prune_data_target_list(data_target_list)

      data_array = np.array(data_target_list[0])  
#      print 'data_array'
   #   pprint.pprint(data_array)
      target_array = np.array(data_target_list[1])
#      print 'target array'
   #   pprint.pprint(target_array)
      cross_validate(data_array, target_array)

def setup_db_cursor(db_name):
   conn = sqlite3.connect(db_name)
   return conn.cursor(), conn

IGNORE_SETS = 

#old ignore sets
#[1, 2, 5, 8, 13, 14, 15, 16, 17, 24, 56, 57,
#58,59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 81, 117,
#      194, 195, 201]
def get_moments(cur, conn, exercise_id):
   #delete ignored sets from db
   for set_id in IGNORE_SETS:
      cur.execute('delete from set_table where _id = ?;', (set_id,)) 
   print 'conn.total_changes after del: '+str(conn.total_changes)

   #retrieve rest of sets for particular exercise from db
   cur.execute('select s.exercise_id, r._id, r.category, m.timestamp, m.euler_angle_X, m.euler_angle_Y, m.euler_angle_Z, m.lin_acc_X, m.lin_acc_Y, m.lin_acc_Z, w.athlete_id from workout_table w inner join set_table s on w._id = s.workout_id inner join rep_table r on s._id = r.set_id inner join moment_table m on r._id = m.rep_id where s.exercise_id = ?;', (exercise_id,))
   moments = cur.fetchall()
   return moments 

def get_athletes(cur, conn):
   cur.execute('select * from athlete_table;')
   athletes = cur.fetchall()
   return athletes

def leave_one_out(athletes, moments):
   results = []
   for athlete in athletes:
      print 'athlete: '+str(athlete[0])
      current_moments = moments
      athlete_moments = []
      
      #separate particular athlete's rep moments from all moments
      for moment in list(current_moments):
         if moment[10] == athlete[0]:
            athlete_moments.append(current_moments.pop(
               current_moments.index(moment)))

#      print 'athlete_moments'
#      if athlete[0] == 1:
#         pprint.pprint(athlete_moments)

      if len(athlete_moments) > 0:

         train_data_target_list = get_data_target_list(current_moments)
         train_data_target_list = prune_data_target_list(train_data_target_list)

         X_train = np.array(train_data_target_list[0])  
         y_train = np.array(train_data_target_list[1])

         test_data_target_list = get_data_target_list(athlete_moments)
         test_data_target_list = prune_data_target_list(test_data_target_list)
         X_test = np.array(test_data_target_list[0])  
         y_test = np.array(test_data_target_list[1])
         
         #X_train.shape
         #y_train.shape
         #X_test.shape
         #y_test.shape

         #make classifier based on rest of moments
         clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,
               min_samples_split=1, random_state=0, n_jobs=-1, criterion='entropy')

         #predict on left out athlete's rep moments
         if len(X_train) > 0 and len(X_train) == len(y_train) and len(X_test) > 0:
            #store prediciton results from each rep 
            result = predict_and_compare(X_train, y_train, X_test, y_test, clf)
            results.append(result)

   return get_accuracy(results)
            
def predict_and_compare(X_train, y_train, X_test, y_test, clf):

   y_pred = clf.fit(X_train, y_train).predict(X_test)

   result = []
   #compare y_pred and y_test
   if len(y_pred) > 0 and len(y_pred) == len(y_test):
      for i, x in enumerate(y_pred):
         print 'y_pred['+str(i)+'] = '+y_pred[i] + ', y_test['+str(i)+'] = '+y_test[i]
         if y_pred[i] == y_test[i]:
            result.append(1)
         else:
            result.append(0)
   else:
      print 'something\'s not right... y_pred and y_test aren\'t the same len'

   return result

def get_accuracy(results):
   avgs = []
   for result in results:
      result_sum = 0
      for is_same in result:
         result_sum += is_same
      avgs.append(result_sum * 1.0 / len(result))

   pprint.pprint(avgs)

   overall_avg = 0
   for avg in avgs:
      overall_avg += avg

   overall_avg /= 1.0 * len(avgs)

   return overall_avg

def rms(x, axis=None):
   return np.sqrt(np.mean(x**2, axis=axis))

def min_max_diff(x):
   y = np.absolute(x)
   max = np.amax(y)
   min = np.amin(y)
   return max - min

measure_index_dict = {'time' : 3, 'ow' : 4, 'ox' : 5, 'oy' : 6, 'oz' : 7, 'lx' : 8, 'ly' : 9,
      'lz' : 10, 'cgx' : 11, 'cgy' : 12, 'cgz' : 13, 'cax' : 14, 'cay' : 15, 'caz' : 16,
      'ccx' : 17, 'ccy' : 18, 'ccz' : 19}
dimension_list = range(3,20) 
#mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness
#feature_function_dict = {'mean' : np.mean, 'var' : np.var, 'std' : np.std, 
#      'max' : np.amax, 'min' : np.amin, 'rms' : rms, 
#      'kurtosis' : sp.stats.kurtosis, 'skew' : sp.stats.skew} 



feature_function_list = [np.mean, np.var, np.std, np.amax, np.amin, rms, 
      sp.stats.kurtosis, sp.stats.skew, min_max_diff] 

def get_data_target_list(moments):
   data_target_list = [[], []]

   rep_id_label_tuple = get_rep_id_label_tuple(moments) 

   for repIdx, rep_id in enumerate(rep_id_label_tuple[0]):

      #make target part of data_target_list
      target_list = rep_id_label_tuple[1][repIdx].split('-')[:-1] 

      #make data part of data_target_list
      rep_dimensions_list = get_rep_dimensions_list()
      for moment in moments:
         if moment[1] == rep_id:
            for index, dimension in enumerate(dimension_list):
               rep_dimensions_list[index].append(moment[dimension])
#      print 'rep_dimensions_list'
#      pprint.pprint(rep_dimensions_list)
      rep_feature_set = []
      for col in rep_dimensions_list:
         col_feature_set = get_feature_set(col)
         #print 'col_feature_set'
         #pprint.pprint(col_feature_set)
         for feature in col_feature_set:
            rep_feature_set.append(feature)


      data_target_list[0].append(rep_feature_set)
      if len(target_list) > 0:
         data_target_list[1].append(target_list[0]) # only works for single labels for now
      else:
         data_target_list[1].append('Correct')

   return data_target_list

#removes labels that are only used once and 'did not complete rep' labels
def prune_data_target_list(data_target_list):
   data_list = data_target_list[0]
   target_list = data_target_list[1]

   target_count = {}

   for target in target_list:
      if target not in target_count:
         target_count[target] = 0
      target_count[target] += 1
#   print 'target_count'
#   pprint.pprint(target_count)
   for target in target_count.keys():
      if target_count[target] < 2:
         #         print 'remove target '+target
         index = target_list.index(target)
         target_list.remove(target)
         data_list.pop(index)
#   print 'target_list'
#   pprint.pprint(target_list)

#throw out did not complete reps
   for target in list(target_list):
#      print 'target'
#      pprint.pprint(target)
      if 'Did not' in target:
         i = target_list.index(target)
         target_list.pop(i)
         data_list.pop(i)
         

   pruned_list = [data_list, target_list]
#   print 'pruned list'
#   pprint.pprint(pruned_list)
   return pruned_list

def get_rep_id_label_tuple(moments):
   rep_ids = []
   rep_labels = []
   for moment in moments:
      current_id = moment[1]
      if current_id not in rep_ids:
         rep_ids.append(current_id)
         rep_labels.append(moment[2])
   return (rep_ids, rep_labels)

def get_rep_dimensions_list():
   dimensions_list = []
   for dimension in measure_index_dict:
      dimensions_list.append([])
   return dimensions_list

def get_feature_set(col):
   feature_set = [] 
   a = np.array(col, dtype=np.float)

   #features based on original data values, acceleration and orientation
   for function in feature_function_list:
      feature_set.append(function(a))

#   b = np.gradient(a)
   
#   for function in feature_function_list:
#      feature_set.append(function(b))

#   c = np.gradient(b)
   
#   for function in feature_function_list:
#      feature_set.append(function(c))

   #d = integrate(a)

   return feature_set

def integrate(a):
   result = np.array()

show_graphs = False 
def cross_validate(data_array, target_array):
   # Create the RFE object and compute a cross-validated score.
   X = data_array
   y = target_array

#   print 'target_array'
#   pprint.pprint(target_array)


#   clf0 = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
#   scores = cross_val_score(clf0, X, y, cv = 10)
#   print 'DecisionTreeClassifier scores.mean(): '+str(scores.mean()) 

   clf1 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
   scores = cross_val_score(clf1, X, y, cv = 10)
   print 'RandomForestClassifier scores.mean(): '+str(scores.mean())                         
   clf2 = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=
         1, random_state=0, n_jobs=-1, criterion='entropy')
   scores = cross_val_score(clf2, X, y, cv = 10)
#   print 'ExtraTreesClassifier scores:'
#   pprint.pprint(scores)
   print 'ExtraTreesClassifier scores.mean(): '+str(scores.mean())
   
   scores = cross_val_score(clf2, X, y, scoring='precision', cv = 10)
   print 'ExtraTreesClassifier precision: '+str(scores.mean())
   pprint.pprint(scores)

#   scores = cross_val_score(clf2, X, y, scoring='average_precision')
#   print 'ExtraTreesClassifier average precision: '+str(scores.mean())
   scores = cross_val_score(clf2, X, y, scoring='recall', cv = 10)
   print 'ExtraTreesClassifier recall: '+str(scores.mean())
   pprint.pprint(scores)
#   s = pickle.dumps(clf2)
#   clf2 = pickle.loads(s)
#   clf2.predict(X[0])

#   clf3 = AdaBoostClassifier(n_estimators=10)
#   scores = cross_val_score(clf3, X, y, cv = 10)
#   print 'AdaBoostClassifier scores.mean(): '+str(scores.mean())

#   clf4 = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
#              max_depth=1, random_state=0)
#   scores = cross_val_score(clf4, X, y, cv = 10)
#   print 'GradientBoostingClassifier scores.mean(): '+str(scores.mean())

   C = 1.5
#   clf5 = SVC(kernel="linear", C=C)
#   scores = cross_val_score(clf5, X, y, cv = 10)
#   print 'SVC linear scores.mean(): '+str(scores.mean())

#  clf6 = SVC(kernel='rbf', gamma=0.7, C=C)
#   scores = cross_val_score(clf6, X, y, cv = 10)
#   print 'SVC rbf scores.mean(): '+str(scores.mean())

#   clf7 = SVC(kernel='poly', degree=3, C=C) 
#   scores = cross_val_score(clf7, X, y, cv = 10)
#   print 'SVC poly scores.mean(): '+str(scores.mean())

#   clf8 = LinearSVC(C=C) 
#   scores = cross_val_score(clf8, X, y, cv = 10)
#   print 'LinearSVC scores.mean(): '+str(scores.mean())

#   clf9 = GaussianNB()
#   scores = cross_val_score(clf9, X, y, cv = 10)
#   print 'GaussianNB scores.mean(): '+str(scores.mean())

#   clf10 = MultinomialNB() 
#   scores = cross_val_score(clf10, X, y, cv = 10)
#   print 'MultinomialNB scores.mean(): '+str(scores.mean())

#   clf11 = BernoulliNB()
#   scores = cross_val_score(clf11, X, y, cv = 10)
#   print 'BernoulliNB scores.mean(): '+str(scores.mean())

#   clf12 = KNeighborsClassifier() 
#   scores = cross_val_score(clf12, X, y, cv = 10)
#   print 'KNeighborsClassifier scores.mean(): '+str(scores.mean())


#   clf13 = SGDClassifier(loss="hinge", penalty="l2") 
#   scores = cross_val_score(clf13, X, y, cv = 10)
#   print 'SGDClassifier scores.mean(): '+str(scores.mean())
   
   clf_list = [clf2] #clf0, clf1

   for clf in clf_list:
      # Split the data into a training set and a test set
      X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
      y_pred = clf.fit(X_train, y_train).predict(X_test)

#         print(classification_report(y_true, y_pred, target_names=target_names))
      
      # Compute confusion matrix
      cm = confusion_matrix(y_test, y_pred)
      print(cm)

   if show_graphs:
      # Show confusion matrix in a separate window
      pl.matshow(cm)
      pl.title('Confusion matrix')
      pl.colorbar()
      pl.ylabel('True label')
      pl.xlabel('Predicted label')
      pl.show()

      #svc = SVC(kernel="linear")
      #rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
      #rfe.fit(X, y)
      #ranking = rfe.ranking_
      #print 'ranking'
      #pprint.pprint(ranking)

      # Plot pixel ranking
#         import pylab as pl
      #pl.matshow(ranking)
      #pl.colorbar()
      #pl.title("Ranking of pixels with RFE")
      #pl.show()
   #   rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
   #                       scoring='accuracy')
   #   rfecv.fit(X, y)

   #   print("Optimal number of features : %d" % rfecv.n_features_)

   #   pl.figure()
   #   pl.xlabel("Number of features selected")
   #   pl.ylabel("Cross validation score (nb of misclassifications)")
   #   pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
   #   pl.show()
      



if __name__ == '__main__':
   main(sys.argv)
