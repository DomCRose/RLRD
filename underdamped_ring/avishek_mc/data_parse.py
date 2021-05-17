import pandas as pd
from matplotlib import pyplot as plt

def convert_csv(fname):
	# Subtract .txt suffix and replace with .csv
	file_csv = fname.split('.txt')[0]+'_data.csv'
	# Open text file for reading and CSV file for writing
	f = open(fname)
	f2 = open(file_csv,'w')
	# Write data from text file to the csv file
	f2.write(f.read())
	# Close both files
	f.close()
	f2.close()
    
def parse_file(fname):
	# Read data from CSV file, assuming tabs as separators
	return pd.read_csv(fname, delimiter='\t', header = None)

#convert_csv("underdamped_v-dependent_forces.txt")
df = parse_file("underdamped_v-dependent_forces_data.csv")
df2 = df.reindex(columns = [0, 11, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5,
							   22, 17, 18, 19, 20, 21, 12, 13, 14, 15, 16])
print(df.values.shape)
print(df.values[0])
print(df.values.T[0])
print(df.values.T[1])
print(df2)
print(df2.values[0])
print(df2.values.T[0])
print(df2.values.T[1])