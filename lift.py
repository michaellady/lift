import sqlite3
import pprint

conn = sqlite3.connect('db.sqlite')
cur = conn.cursor()

cur.execute('SELECT * FROM rep_table')

reps = cur.fetchall()

pprint.pprint(reps)

moment = {}
for rep in reps:
   print('type(rep[2]) ', str(type(rep[2])))
   cur.execute('SELECT * FROM moment_table WHERE rep_id = ?', (str(rep[2]),))
   moment[rep[0]] = cur.fetchall()

pprint.pprint(moment)
