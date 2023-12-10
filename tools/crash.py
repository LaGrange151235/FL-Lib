import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()
path = args.path

for dir in os.listdir(path):
    file_path = path+"/"+dir
    if "client" in str(dir):
        for line in open(file_path):
            if "Traceback" in line:
                print("error at", dir)
                exit(0)