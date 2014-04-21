import sys
import sqlite3
import pprint
import scipy as sp
from scipy import stats
import numpy as np

exercises = {'press': 0, 'bench' : 1, 'squat' : 2}

def main(argv):
   cur = setup_db_cursor(argv[1])
   sets = get_exercise_sets(cur, 1)
   print 'num sets: '+str(len(sets))
   set_rep_dict = get_set_reps(cur, sets)
   print 'set_rep_dict'
   pprint.pprint(set_rep_dict)
   rep_moment_dict = get_rep_moments(cur, set_rep_dict.values())
   rep_features_dict = get_rep_features(rep_moment_dict)
   data_array = get_data_array(rep_features_dict)
#   pprint.pprint(data_array)

def setup_db_cursor(db_name):
   conn = sqlite3.connect('db.sqlite')
   return conn.cursor()

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
      print 'set: '+str(set[0])
      pprint.pprint(set_rep_dict[set])

#   pprint.pprint(set_rep_dict)
   return set_rep_dict 

#get all of the moments that map to a given rep
def get_rep_moments(cur, reps):
   rep_moment_dict = {}
   #reps has an extra array enclosing it
   print 'reps'
   pprint.pprint(reps)
   for rep in reps:
      #      pprint.pprint(reps)
#      pprint.pprint(rep)
      cur.execute('SELECT * FROM moment_table WHERE rep_id = ?', (rep[0],))
      rep_moment_dict[rep[0]] = cur.fetchall()

   #pprint.pprint(rep_moment_dict)
   return rep_moment_dict

#get features of each rep based on moments
def get_rep_features(rep_moment_dict):
   rep_feature_dict = {}
   for rep in rep_moment_dict.keys():
      rep_feature_dict[rep] = get_feature_set(rep_moment_dict[rep])

   return rep_feature_dict

def rms(x, axis=None):
       return np.sqrt(np.mean(x**2, axis=axis))

#orientation index: 3, 4, 5 linacc index: 6, 7, 8
measure_index_dict = {'ox' : 3, 'oy' : 4, 'oz' : 5, 'lx' : 6, 'ly' : 7,
      'lz' : 8}
#mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness
feature_function_dict = {'mean' : np.mean, 'var' : np.var, 'std' : np.std, 
   'max' : np.amax, 'min' : np.amin, 'rms' : rms, 
   'kurtosis' : sp.stats.kurtosis, 'skew' : sp.stats.skew} 

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
   print 'len(rep_features_dict.keys()): '+str(len(rep_features_dict.keys()))
   for rep in rep_features_dict.keys():
      data_array.append([])
      for dimension in rep_features_dict[rep].keys():
         for function in rep_features_dict[rep][dimension].keys():
            data_array[len(data_array) - 1].append(rep_features_dict[rep][dimension][function])
   return data_array

if __name__ == '__main__':
   main(sys.argv)
