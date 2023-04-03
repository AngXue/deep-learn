import os

allNum = 0
for path in os.listdir("trainPhotos"):
    allNum += len(os.listdir("trainPhotos/" + path))
print(allNum)
