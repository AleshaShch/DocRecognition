import sys
import argparse
from unittest import TestCase
from doc_recognition import DocRecognition


class TestDocRecognition(TestCase):
    def test_credit_card_recognition_time(self):
        parser = argparse.ArgumentParser(prog=sys.argv[0])
        parser.add_argument("-i", "--image", required=True, help="Add Path to input image")
        parser.add_argument("-r", "--report", required=False, help="Add Path to report folder", default=0)
        args = vars(parser.parse_args())
        doc_recognition = DocRecognition()
        doc_recognition.upload_image(args["image"])
        DocRecognition.credit_card_recognition()
        DocRecognition.get_recognition_result()
        TestCase.assert_
