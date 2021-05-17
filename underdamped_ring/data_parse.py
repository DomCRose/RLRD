import pandas as pd
from matplotlib import pyplot as plt

def convert_csv(fname):
	# Subtract .txt suffix and replace with .csv
	file_csv = fname.split('.txt')[0]+'.csv'
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

#convert_csv("underdamped_cumulant_expansion_data.txt")
df = parse_file("underdamped_cumulant_expansion_data.csv")
print(df.values.shape)

plt.plot(df.values.T[0], df.values.T[1], label = '1')
plt.plot(df.values.T[0], df.values.T[2], label = '2')
plt.plot(df.values.T[0], df.values.T[3], label = '3')
plt.plot(df.values.T[0], df.values.T[4], label = '4')
plt.plot(df.values.T[0], df.values.T[5], label = '5')
plt.legend()
plt.show()