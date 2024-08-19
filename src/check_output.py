from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='gpt4', type=str)
args = parser.parse_args()

x = load_file(args.path)
print(x)