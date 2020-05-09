import utilities as utils
import time 
import sys 

system = sys.argv[1]

prev_time = time.time()
for i in range(20):
    time.sleep(.5)
    if i % 4 == 0:
        prev_time = utils.eta_counter(20, i, prev_time, every=4, system=system)
    
    
