
echo SQUAT
python oldlift.py squat ./db/v1.sqlite -o 

echo BENCH
python oldlift.py bench ./db/v1.sqlite -o

echo PRESS
python oldlift.py press ./db/v1.sqlite -o
