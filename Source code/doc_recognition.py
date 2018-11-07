import sys
import argparse
import cv2
import credit_card


class DocRecognition:
    def __init__(self, image=None):
        self.__set_image(image)
        self._credit_card = credit_card.CreditCard()

    def __set_image(self, image):
        self.__image = image

    def __get_image(self):
        return self.__image

    image = property(__get_image, __set_image)

    def upload_image(self, image):
        self.__set_image(image)

    def credit_card_recognition(self):
        pass

    def get_recognition_result(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="DocRecognition allows you to get information from "
                                                                   "document images")
    parser.add_argument("-i", "--image", required=True, help="Add Path to input image")
    parser.add_argument("-r", "--report", required=False, help="Add Path to report folder", default=0)
    args = vars(parser.parse_args())
    doc_recognition = DocRecognition()
    doc_recognition.upload_image(cv2.imread(args["image"]))
