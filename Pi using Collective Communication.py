from __future__ import division
from decimal import *
import math
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

total_sum = 0

def aggregator(precision):

    partial_sum = 0
    
    global total_sum
    for i in range(rank,precision+1,size):
        
        temp1 = (1 / (16**i))
        temp2 = (4 / ((8 * i) + 1))
        temp3 = (2 / ((8 * i) + 4))
        temp4 = (1 / ((8 * i) + 5))
        temp5 = (1 / ((8 * i) + 6))
        
        intermediate = temp2-temp3-temp4-temp5
        
        final = Decimal((temp1)*(intermediate))
        
        partial_sum += final
    
    total_sum = comm.reduce(partial_sum,root=0,op=MPI.SUM)    
    
if __name__=="__main__":
    
    precision = 1000
    
    start_time = MPI.Wtime()
     
    getcontext().prec = 1000        # Setting the precision to 1000 since we need 1001 digits
         
    aggregator(precision)
        
    if rank == 0:
        print("The calculated pi value is : \n")
        print(total_sum)
        print("\nTime taken is : "+str(MPI.Wtime()-start_time))
        print("\n\n")
