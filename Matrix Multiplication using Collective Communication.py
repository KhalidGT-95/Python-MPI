import numpy as np
from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def compute(Matrix_A,Matrix_B,Matrix_C,dimension):
    
   for i in range(rank,dimension,size):         
       for k in range(0,dimension):
           for l in range(0,dimension):
               Matrix_C[i][k] += Matrix_A[i][l] * Matrix_B[l][k]                  
              
   return Matrix_C
        
if __name__=="__main__":

    dimension = [100,500,1000]      # Different size for matrix
        
    for i in range(0,3):            # Running for each size
        if rank == 0:
           
            print("\nRunning for matrix size : "+str(dimension[i])+"x"+str(dimension[i]))
        
        np.random.seed(111)
        
        Matrix_C = np.zeros([dimension[i],dimension[i]],int)                # Result stored in this matrix
        
        Matrix_A = np.random.randint(5,size=(dimension[i],dimension[i]))    # Matrix A
        Matrix_B = np.random.randint(5,size=(dimension[i],dimension[i]))    # Matrix B
           
        start_time = MPI.Wtime()
        
        data = compute(Matrix_A,Matrix_B,Matrix_C,dimension[i])
        
        Matrix_C = comm.reduce(data,op=MPI.SUM,root=0)
        
        comm.Barrier()              # make sure that all processes have reached a stable state
        
        if rank == 0:
            print("\nThe total time taken is : " + str(MPI.Wtime() - start_time)+"\n\n")
