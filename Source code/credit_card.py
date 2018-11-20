import numpy
import cv2
from imutils import contours as cntrs


class CreditCard:
    OCR_THRESH_VALUE = 30
    OCR_THRESH_MAXVALUE = 255
    CHARACTER_WIDTH = 22
    CHARACTER_HIGH = 34

    def __init__(self):
        self.__recognition_report = []
        self.__source_image = None

    def __set_ocr_image(self, image):
        self.__ocr_image = image

    def __get_ocr_image(self):
        return self.__ocr_image

    def __add_file_to_report(self, image):
        self.__recognition_report.append(image)

    def __draw_all_contours(self, image, contours):
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        return image

    ocr_image = property(__get_ocr_image)

    def get_template_characters(self):
        self.__set_ocr_image(cv2.imread("OCR-A_1.png"))
        ocr_image = self.ocr_image_filtering(self.__get_ocr_image())
        characters_contours = cv2.findContours(ocr_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        characters_contours = cntrs.sort_contours(characters_contours, method="left-to-right")[0]

        characters = {}
        for (i, c) in enumerate(characters_contours):
            (x, y, w, h) = cv2.boundingRect(c)
            characters[i] = ocr_image[y:y + h, x:x + w]
            characters[i] = cv2.resize(characters[i], (self.CHARACTER_WIDTH, self.CHARACTER_HIGH))
        return characters

    def ocr_image_filtering(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(image, self.OCR_THRESH_VALUE, self.OCR_THRESH_MAXVALUE, cv2.THRESH_BINARY_INV)[1]
        return image

    def image_filtering(self, image):
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))
        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        self.__source_image = image
        self.__add_file_to_report(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.__add_file_to_report(image)
        image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, rect_kernel)
        self.__add_file_to_report(image)

        grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = numpy.absolute(grad_x)
        (min_value, max_value) = (numpy.min(grad_x), numpy.max(grad_x))
        grad_x = (255 * ((grad_x - min_value) / (max_value - min_value)))
        grad_x = grad_x.astype("uint8")

        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        self.__add_file_to_report(grad_x)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
        self.__add_file_to_report(thresh)
        return thresh

    def find_informational_fields(self, image):
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours = cntrs.sort_contours(contours, method="top-to-bottom")[0]
        self.__add_file_to_report(self.__draw_all_contours(self.__source_image.copy(), contours))
        return contours

    def split_information_fiels(self, contours):
        fields = []
        y_field_number = 300

        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = w / float(h)
            if 2.5 < ratio < 4.0:
                if 40 < w < 60 and 10 < h < 20:
                    fields.append((x, y, w, h))
                    y_field_number = y
                    if len(fields) == 4:
                        fields.sort()
            elif ratio > 3 and y > y_field_number and x < fields[2][0] + fields[2][2]:
                fields.append((x, y, w, h))

        fields = sorted(fields, key=lambda contour: contour[1])
        return fields

    def recognize_characters(self, fields, source_image):
        characters = []
        image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        template_characters = self.get_template_characters()

        for (i, (x, y, w, h)) in enumerate(fields):
            character_group = image[y - 5: y + h + 5, x - 5: x + w + 5]
            character_group = cv2.threshold(character_group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            character_contours = cv2.findContours(character_group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            character_contours = cntrs.sort_contours(character_contours, method="left-to-right")[0]

            for c in character_contours:
                recognize_group = []
                (character_x, character_y, character_w, character_h) = cv2.boundingRect(c)
                character_roi = character_group[character_y: character_y + character_h, character_x: character_x +
                                                character_w]
                character_roi = cv2.resize(character_roi, (self.CHARACTER_WIDTH, self.CHARACTER_HIGH))
                matches = []
                for (character, character_template) in template_characters.items():
                    result = cv2.matchTemplate(character_roi, character_template, cv2.TM_CCOEFF)
                    (_, coincidence, _, _) = cv2.minMaxLoc(result)
                    matches.append(coincidence)

                recognize_group.append(numpy.argmax(matches))
                char = recognize_group.pop()
                if char > 9:
                    characters.extend(chr(char + 55))
                else:
                    characters.extend(str(char))

        information = {'number': characters[0:16], 'name': characters[16:]}
        return information

    def get_recognition_report(self):
        return self.__recognition_report
