import sys
sys.path.append('/vepfs/wcf/G/zekai/root/.code/LearnRepr/src')
from utils import *

APIs_description = load_file(f'/vepfs/wcf/G/zekai/root/.code/LearnRepr/data/tensorflow_icl2/api_desc.pkl')

save_file(APIs_description, '/vepfs/wcf/G/zekai/root/.code/LearnRepr/data/tensorflow_icl2/APIs_description.json')