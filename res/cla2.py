import os
import shutil


def classification_image():
    Image_dir = os.path.join(os.getcwd() + "\\" + "trainPhotos")
    # print(Image_dir)

    train_name = os.path.join(os.getcwd() + "\\" + "trainPhotos")
    test_name = os.path.join(os.getcwd() + "\\" + "testPhotos")

    if os.path.exists(train_name):
        print(f"{train_name}文件夹已存在")
    else:
        os.makedirs(train_name)

    if os.path.exists(test_name):
        print(f"{test_name}文件夹已存在")
    else:
        os.makedirs(test_name)

    files = os.listdir(train_name)

    for image_location in files:
        # people=os.path.join(train_name+"\\"+image_location)

        target_train = os.path.join(train_name + "\\" + image_location)
        target_test = os.path.join(test_name + "\\" + image_location)

        name = os.listdir(target_train)
        # print(name)

        l = len(name)
        j = 1

        for i in name:
            if j == l and j != 1:
                leni = len(i)
                shutil.copy(os.path.join(target_train + "\\" + i),
                            os.path.join(test_name + "\\" + i[:leni - 9] + i[leni - 4:]))
                os.remove(os.path.join(target_train + "\\" + i))

            j += 1


classification_image()
