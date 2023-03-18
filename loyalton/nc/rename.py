import os

files = os.listdir('.')

print(files)

for file in files:
    newfile = file.replace(":", "_")
    os.rename(file, newfile)