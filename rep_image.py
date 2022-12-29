from PIL import Image
import cv2


def gauss_filter(path):
    """Добавляет гауссовское размытие"""
    my_photo = cv2.imread(path)
    gaussian_image  = cv2.GaussianBlur(my_photo,(7, 7), 0)
    cv2.imwrite(path[:len(path) - 4] + "_gauss.png", gaussian_image)
    repair(path[:len(path) - 4] + "_gauss.png")


def repair(path):
    """Увеличивает контрастность и делает ярковыраженным края"""
    with Image.open(path) as img:
        img.load()
        gray = img.convert('L')
        threshold = 100
        img_threshold = gray.point(lambda x: 255 if x > threshold else 0)
        img_threshold.save(path[:len(path)-10] + '_rep.png')