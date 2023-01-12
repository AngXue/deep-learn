# coding:utf-8
import cv2

# 临时测试程序

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    flag = cap.isOpened()

    index = 1
    while flag:
        ret, frame = cap.read()
        cv2.imshow("Capture", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):  # 按下s键，进入下面的保存图片操作
            cv2.imwrite("../res/" + str(index) + ".jpg", frame)
            print(cap.get(3))
            print(cap.get(4))
            print("save" + str(index) + ".jpg successfully!")
            print("-------------------------")
            index += 1
        elif k == ord('q'):  # 按下q键，程序退出
            break
    cap.release()
    cv2.destroyAllWindows()
