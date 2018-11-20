import sys
import os
import argparse
import cv2
import imutils
import credit_card


class DocRecognition:
    IMAGE_WIDTH = 300

    def __init__(self, image=None):
        self.__set_image(image)
        self._credit_card = credit_card.CreditCard()
        self.__recognition_result = None
        self.__recognition_report = None

    def __set_image(self, image):
        self.__image = image

    def __get_image(self):
        return self.__image

    image = property(__get_image, __set_image)

    def upload_image(self, image):
        self.__set_image(imutils.resize(image=image, width=self.IMAGE_WIDTH))

    def credit_card_recognition(self):
        image = self._credit_card.image_filtering(self.__image)
        contours = self._credit_card.find_informational_fields(image)
        fields = self._credit_card.split_information_fiels(contours)
        self.__recognition_result = self._credit_card.recognize_characters(fields, self.__image)

    def get_recognition_result(self, arguments):
        print("Credit card number: {}".format("".join(self.__recognition_result['number'])))
        print("Card holder name: {}".format("".join(self.__recognition_result['name'])))
        if arguments["report"]:
            self.__recognition_report = self._credit_card.get_recognition_report()
            j = 0
            for i in self.__recognition_report:
                cv2.imwrite(os.path.join(arguments["report"], 'image{}.jpg'.format(j)), i)
                j += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="DocRecognition allows you to get information from "
                                                                   "document images")
    parser.add_argument("-i", "--image", required=True, help="Add Path to input image")
    parser.add_argument("-r", "--report", required=False, help="Add Path to report folder", default=0)
    args = vars(parser.parse_args())
    doc_recognition = DocRecognition()
    doc_recognition.upload_image(cv2.imread(args["image"]))
    doc_recognition.credit_card_recognition()
    doc_recognition.get_recognition_result(args)
