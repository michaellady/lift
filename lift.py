import sys
import sqlite3
import pprint
import scipy as sp
from scipy import stats
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix

exercises = {'press': 0, 'bench' : 1, 'squat' : 2}

# sql mega query select * from set_table s inner join rep_table r on s._id = r.set_id inner join moment_table m on r._id = m.rep_id where exercise_id = ?;

def main(argv):
   cur, conn = setup_db_cursor(argv[1])
   
#   sets = get_exercise_sets(cur, 1)
#   print 'num sets: '+str(len(sets))
#   set_rep_dict = get_set_reps(cur, sets)
#   print 'set_rep_dict'
#   pprint.pprint(set_rep_dict)
#   rep_moment_dict = get_rep_moments(cur, set_rep_dict.values())
#   print 'rep_moment_dict'
#   pprint.pprint(rep_moment_dict)

#   rep_features_dict = get_rep_features(rep_moment_dict)
#   data_array = get_data_array(rep_features_dict)
#   pprint.pprint(data_array)
#   print 'len(data_array): '+str(len(data_array))
   
#   target_array = get_target_array(set_rep_dict)
#   print 'len(target_array): '+str(len(target_array))
 #  pprint.pprint(target_array)

   moments = get_moments(cur, conn, exercises['press'])
   print 'moments'
#   pprint.pprint(moments)
   print 'len moments: '+str(len(moments))

   data_target_list = get_data_target_list(moments)
   data_target_list = prune_data_target_list(data_target_list)

   data_array = np.array(data_target_list[0])  
   print 'data_array'
#   pprint.pprint(data_array)
   target_array = np.array(data_target_list[1])
   print 'target array'
#   pprint.pprint(target_array)
   classify(data_array, target_array)

def setup_db_cursor(db_name):
   conn = sqlite3.connect(db_name)
   return conn.cursor(), conn

IGNORE_SETS = [1, 2, 5, 8, 13, 14, 15, 16, 17, 24]
def get_moments(cur, conn, exercise_id):
   #delete ignored sets from db
   for set_id in IGNORE_SETS:
      cur.execute('delete from set_table where _id = ?;', (set_id,)) 
   print 'conn.total_changes after del: '+str(conn.total_changes)

   #retrieve rest of sets for particular exercise from db
   cur.execute('select s.exercise_id, r._id, r.category, m.timestamp, m.euler_angle_X, m.euler_angle_Y, m.euler_angle_Z, m.lin_acc_X, m.lin_acc_Y, m.lin_acc_Z  from set_table s inner join rep_table r on s._id = r.set_id inner join moment_table m on r._id = m.rep_id where s.exercise_id = ?;', (exercise_id,))
   moments = cur.fetchall()
   return moments 


#get all of the sets that map to a given exercise
def get_exercise_sets(cur, exercise_id):
   cur.execute('SELECT * FROM set_table WHERE exercise_id = ?', (exercise_id,))
   sets = cur.fetchall()
   return sets

#get all of the sets for all exercises
def get_all_sets(cur):
   exercise_set_dict = {}
   for exercise_id in exercises.values():
      exercise_set_dict[exercise_id] = get_exercise_sets(cur, exercise_id)
   return exercise_set_dict 

#get all of the reps that map to a given set
def get_set_reps(cur, sets):
   set_rep_dict = {}
   for set in sets:
      cur.execute('SELECT * FROM rep_table WHERE set_id = ?', (set[0],))
      set_rep_dict[set] = cur.fetchall()
#      print 'set: '+str(set[0])
#      pprint.pprint(set_rep_dict[set])

#   pprint.pprint(set_rep_dict)
   return set_rep_dict 

#get all of the moments that map to a given rep
def get_rep_moments(cur, sets):
   rep_moment_dict = {}
   #reps has an extra array enclosing it
#   print 'sets'
#   pprint.pprint(sets)
   for set in sets:
#      print 'set'
#      pprint.pprint(set)
      for rep in set:
#         print 'rep'
#         pprint.pprint(rep)
         cur.execute('SELECT * FROM moment_table WHERE rep_id = ?', (rep[0],))
         rep_moment_dict[rep[0]] = cur.fetchall()

   #pprint.pprint(rep_moment_dict)
   return rep_moment_dict

#get features of each rep based on moments
def get_rep_features(rep_moment_dict):
   rep_feature_dict = {}
   for rep in rep_moment_dict.keys():
         if len(rep_moment_dict[rep]) > 0:
            rep_feature_dict[rep] = get_feature_set(rep_moment_dict[rep])

   return rep_feature_dict

def rms(x, axis=None):
       return np.sqrt(np.mean(x**2, axis=axis))

#orientation index: 3, 4, 5 linacc index: 6, 7, 8
measure_index_dict = {'ox' : 4, 'oy' : 5, 'oz' : 6, 'lx' : 7, 'ly' : 8,
      'lz' : 9}
dimension_list = range(4,10) 
#mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness
feature_function_dict = {'mean' : np.mean, 'var' : np.var, 'std' : np.std, 
   'max' : np.amax, 'min' : np.amin, 'rms' : rms, 
   'kurtosis' : sp.stats.kurtosis, 'skew' : sp.stats.skew} 

feature_function_list = [np.mean, np.var, np.std, np.amax, np.amin, rms, 
      sp.stats.kurtosis, sp.stats.skew] 

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
      data_target_list[1].append(target_list[0]) # only works for single labels for now
      
   return data_target_list

#removes labels that are only used once
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
   a = np.array(col)
   
   for function in feature_function_list:
      feature_set.append(function(a))

   return feature_set


"""
def get_feature_set(moments):
#   pprint.pprint(moments)
   feature_set_dict = {}
   for dimension in measure_index_dict.keys(): 
      feature_set_dict[dimension] = {}
      idx = measure_index_dict[dimension]
      col = []
      for moment in moments:
         col.append(moment[idx])
#      print('col for dimension '+dimension)
#      pprint.pprint(col)
      a = np.array(col)
      for function in feature_function_dict:
         feature_set_dict[dimension][function] =feature_function_dict[function](a) 
#         print('dimension: '+dimension)
#         pprint.pprint(feature_set_dict[dimension]) 
   return feature_set_dict
   

def get_data_array(rep_features_dict): 
   data_array = []
#   print 'len(rep_features_dict.keys()): '+str(len(rep_features_dict.keys()))
#   pprint.pprint(rep_features_dict)
   for rep in rep_features_dict.keys():
      data_array.append([])
      for dimension in rep_features_dict[rep].keys():
         for function in rep_features_dict[rep][dimension].keys():
            data_array[len(data_array) - 1].append(rep_features_dict[rep][dimension][function])
   return data_array

def get_target_array(set_rep_dict):
   target_array = []
   for set in set_rep_dict:
      for rep in set_rep_dict[set]:
#         pprint.pprint(rep)
         target = rep[len(rep) - 1].split('-')
#         print 'target: '
#         pprint.pprint(target)
         if(len(target[0]) > 0):
            target_array.append(target[0])
   return target_array
"""

def classify(data_array, target_array):
   # Create the RFE object and compute a cross-validated score.
   X = data_array
   y = target_array

   svc = SVC(kernel="linear")
#   rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
#                       scoring='accuracy')
#   rfecv.fit(X, y)

#   print("Optimal number of features : %d" % rfecv.n_features_)
   
#   import pylab as pl
#   pl.figure()
#   pl.xlabel("Number of features selected")
#   pl.ylabel("Cross validation score (nb of misclassifications)")
#   pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#   pl.show()


   clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
   scores = cross_val_score(clf, X, y, cv = 10)
   print 'DecisionTreeClassifier scores.mean(): '+str(scores.mean()) 

   clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
   scores = cross_val_score(clf, X, y, cv = 10)
   print 'RandomForestClassifier scores.mean(): '+str(scores.mean())                         

   clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
   scores = cross_val_score(clf, X, y, cv = 10)
   print 'ExtraTreesClassifier scores.mean(): '+str(scores.mean())

if __name__ == '__main__':
   main(sys.argv)
