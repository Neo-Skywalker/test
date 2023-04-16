MY_RS = 256 #要求MY_RS>=128
RS_LVL = 2
SIGMA_THRESHOLD = 0
Z_S_N = min(MY_RS,128)# 如果小于128，就没必要优化了
'''
python run_nerf.py --mode render --conf confs/nerf.conf --case test
'''