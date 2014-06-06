echo PRESS
python lift.py press ./db/db7.sqlite -o
echo BENCH
python lift.py bench ./db/smartbell_database.db.2 -o
echo SQUAT
python lift.py squat ./db/smartbell_database.db.2 -o
