"""
This module provides an API for documents recognition such as credit card (version 0.1) and passport (version 0.2).
"""

import sys
import os
import argparse
import cv2
import imutils
import credit_card


class DocRecognition:
    """
    Base class that provides the API for documents recognition.
    """

    IMAGE_WIDTH = 300

    def __init__(self, image=None):
        """
        Constructor of the DocRecognition class.
        :param image: source image for document recognition.
        """
        self.__set_image(image)
        self._credit_card = credit_card.CreditCard()
        self.__recognition_result = None
        self.__recognition_report = None

    def __set_image(self, image):
        """
        Set the source image for document recognition.
        :param image: source image.
        """
        self.__image = image

    def __get_image(self):
        """
        Get the source image for document recognition.
        :return: source image.
        """
        return self.__image

    image = property(__get_image, __set_image)

    def upload_image(self, image):
        """
        Upload the source image for document recognition.
        :param image: source image.
        """
        self.__set_image(imutils.resize(image=image, width=self.IMAGE_WIDTH))

    def credit_card_recognition(self):
        """
        Start a document recognition cycle.
        """
        image = self._credit_card.image_filtering(self.__image)
        contours = self._credit_card.find_informational_fields(image)
        fields = self._credit_card.split_informational_fields(contours)
        self.__recognition_result = self._credit_card.recognize_characters(fields, self.__image)

    def get_recognition_result(self, arguments):
        """
        Get results of document recognition.
        :param arguments: parameters of document recognition.
        """
        print("Credit card number: {}".format("".join(self.__recognition_result['number'])))
        print("Card holder name: {}".format("".join(self.__recognition_result['name'])))
        if arguments["report"]:
            self.__recognition_report = self._credit_card.get_recognition_report()
            j = 0
            for i in self.__recognition_report:
                cv2.imwrite(os.path.join(arguments["report"], 'image{}.jpg'.format(j)), i)
                j += 1


if __name__ == '__main__':
    # Create an argument parser for parameters of document recognition
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="DocRecognition allows you to get information from "
                                                                   "document images")
    parser.add_argument("-i", "--image", required=True, help="Add Path to input image")
    parser.add_argument("-r", "--report", required=False, help="Add Path to report folder", default=0)
    args = vars(parser.parse_args())
    doc_recognition = DocRecognition()
    doc_recognition.upload_image(cv2.imread(args["image"]))
    doc_recognition.credit_card_recognition()
    doc_recognition.get_recognition_result(args)