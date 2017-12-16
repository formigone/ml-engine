import argparse
import os
from PIL import Image
import numpy as np


def main(dir, out_file, class_val):
    with open(out_file, 'w+') as fh:
        # fh.write('10,120000\n')
        for file in os.listdir(dir):
            img = Image.open(dir + '/' + file, 'r')
            img = np.array(img.getdata()) / 255
            img = np.char.mod('%f', img.reshape((200 * 200 * 3)))
            fh.write(','.join(img) + ',' + str(class_val) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Directory to parse')
    parser.add_argument('--out_file', type=str, help='Where to save the dataset to')
    parser.add_argument('--class_val', type=int, default=0, help='Class value for this image set')

    FLAGS, unused = parser.parse_known_args()
    main(FLAGS.dir, FLAGS.out_file, FLAGS.class_val)
