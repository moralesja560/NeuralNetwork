# a quick file renamer
import os

# select the class folder. In this case i had pizza pictures in one folder.
folder = r'D:\Downloads\archive\pizza_not_pizza\pizza'
for count, filename in enumerate(os.listdir(folder)):
	dst = f"steak{str(count)}.jpg"
	src =f"{folder}\\{filename}"  # foldername/filename, if .py file is outside folder
	dst =f"{folder}\\{dst}"
		# rename() function will
	# rename all the files
	print(src)
	#os.rename(src, dst)