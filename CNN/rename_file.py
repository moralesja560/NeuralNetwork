# a quick file renamer
import os

# select the class folder. In this case i had pizza pictures in one folder.
folder = r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\resources\dataset_pizza_steak\pizza_steak\steak'
for count, filename in enumerate(os.listdir(folder)):
	dst = f"steak{str(count)}.jpg"
	src =f"{folder}\\{filename}"  # foldername/filename, if .py file is outside folder
	dst =f"{folder}\\{dst}"
		# rename() function will
	# rename all the files
	#print(dst)
	os.rename(src, dst)