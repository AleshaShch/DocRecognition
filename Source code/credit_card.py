import cv2
import imutils
import numpy
from imutils import contours


class CreditCard:
    OCR_THRESH_VALUE = 45
    OCR_THRESH_MAXVALUE = 255

    def __set_ocr_image(self, image):
        self.__ocr_image = image

    def __get_ocr_image(self):
        return self.__ocr_image

    ocr_image = property(__get_ocr_image)

    def get_template_characters(self):
        self.__set_ocr_image(cv2.imread("OCR-B.jpg"))
        ocr_image = self.ocr_image_filtering(self.__get_ocr_image())
        characters_contours = cv2.findContours(ocr_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        characters_contours = contours.sort_contours(characters_contours, method="left-to-right")[0]
        characters = {}
        for (i, c) in enumerate(characters_contours):
            (x, y, w, h) = cv2.boundingRect(c)
            characters[i] = ocr_image[y:y + w, x:x + h]
        return characters

    def ocr_image_filtering(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(image, self.OCR_THRESH_VALUE, self.OCR_THRESH_MAXVALUE, cv2.THRESH_BINARY_INV)[1]
        return image

    def image_filtering(self, image):
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, rect_kernel)

        grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = numpy.absolute(grad_x)
        (min_value, max_value) = (numpy.min(grad_x), numpy.max(grad_x))
        grad_x = (255 * ((grad_x - min_value) / (max_value - min_value)))
        grad_x = grad_x.astype("uint8")

        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
        return thresh
