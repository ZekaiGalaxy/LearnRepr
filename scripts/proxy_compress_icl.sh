# run proxy compress on selected apis
python run_proxy_compress.py --task=tensorflow_icl1 --embedding=gte-small --num_epoch=5 --do_eval=0

# generate descriptions using icl
python icl_gen.py --task=tensorflow

# run proxy compress on all apis
python run_proxy_compress.py --task=tensorflow_icl2