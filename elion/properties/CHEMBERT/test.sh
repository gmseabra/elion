rm -f predictions.dat
python -m  chembert.py
diff -q predictions.dat predictions_orig.dat
