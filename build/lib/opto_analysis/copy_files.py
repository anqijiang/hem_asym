import shutil
import os

# path to source directory
copy_dir = 'C:\\Users\\Sheffield_lab\\PycharmProjects\\opto_analysis'

# path to destination directory
dest_dir = 'C:\\Users\\Sheffield_lab\\PycharmProjects\\Heather'

# getting all the files in the source directory
files = os.listdir(copy_dir)

shutil.copytree(copy_dir, dest_dir)

