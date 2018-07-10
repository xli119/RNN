"""
Author: Xiaoyu Li
Created on 7/10/2018
change files' encoding to utf-8
"""

import os
import codecs
import pickle
from chardet.universaldetector import UniversalDetector
import chardet


def change_encoding(in_path, out_path):
    detector = UniversalDetector()
    f_list = os.listdir(in_path)
    for file in f_list:
        in_name = os.path.join(in_path, file)
        out_name = os.path.join(out_path, file[:-3]+"txt")

        with codecs.open(in_name, 'r') as source_file:
            with codecs.open(out_name, 'w', encoding='utf-8') as target_file:
                contents = source_file.read()

                if not contents:
                    break


                target_file.write(contents)


if __name__ == '__main__':
    change_encoding("D:\\whole\\test", "D:\\whole\\testsave")