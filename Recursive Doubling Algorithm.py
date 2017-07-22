#!/usr/bin/python

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def destA(rank):                    # Calculate destination A 
    temp = int((2 * rank) + 1)
    return temp
        
def destB(rank):                    # Calculate destination A 
    temp = int((2 * rank) + 2)
    return temp
    
def recvProc(rank):                 # Calculate the rank of the process to receive from
    temp = int((rank-1) / 2)
    return temp
    
def sendAll():

    data = 0
    start_time = None
    
    if rank == 0:
        
        size_array = int(input("Enter the size for the array : "))
        
        start_time = MPI.Wtime()
        
        array = np.random.randint(10,size=size_array)  
            
        dest_A = destA(rank)
        dest_B = destB(rank)
        comm.send(array,dest=dest_A,tag=1)
        comm.send(array,dest=dest_B,tag=1)
            
    else:
        recvValue = recvProc(rank)              # Calculate the rank to receive from
        data = comm.recv(source=recvValue)      # First wait for the value to be received
        dest_A = destA(rank)                    
        dest_B = destB(rank)
        if dest_A < size:                       # If destA lies within the number of processes then send the data
            comm.send(data,dest=dest_A,tag=1)
        if dest_B < size:                       # If destB lies within the number of processes then send the data
            comm.send(data,dest=dest_B,tag=1)   
    
    comm.Barrier()
    
    if rank == 0:
        end_time = MPI.Wtime() - start_time
        print("The time taken to send to " + str(size-1) + " processes is : " + str(end_time))        
                    
if __name__=="__main__":
    
    sendAll()
    
