import cv2
import numpy as np
from mpi4py import MPI
import pylab as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Histogram_array_R = np.zeros(256,int)       # Used by each process to store the partial values of red 
Histogram_array_G = np.zeros(256,int)       # Used by each process to store the partial values of green
Histogram_array_B = np.zeros(256,int)       # Used by each process to store the partial values of blue

Final_array_R = np.zeros(256,int)           # Used for collecting partial values from each process and merging them together in a final array for generating histogram for Red
Final_array_G = np.zeros(256,int)           # Used for collecting partial values from each process and merging them together in a final array for generating histogram for Green
Final_array_B = np.zeros(256,int)           # Used for collecting partial values from each process and merging them together in a final array for generating histogram for Blue


def RGBScale(image):
    
    global Histogram_array
    global Final_array
    
    for i in range(rank,len(image),size):
        for j in range(len(image[0])):
            value_b = image[i][j][0]
            value_g = image[i][j][1]
            value_r = image[i][j][2]
            
            Histogram_array_R[value_r] += 1
            Histogram_array_G[value_g] += 1
            Histogram_array_B[value_b] += 1
    
    comm.Reduce(Histogram_array_R,Final_array_R,root=0,op=MPI.SUM)
    comm.Reduce(Histogram_array_G,Final_array_G,root=0,op=MPI.SUM)
    comm.Reduce(Histogram_array_B,Final_array_B,root=0,op=MPI.SUM)
    

if __name__=="__main__":
    
    RGB_img = cv2.imread('2048.jpg')
    
    start = MPI.Wtime()
    
    RGBScale(RGB_img)
    
    if rank == 0:
        
        print("The total time taken by the processes comes out to be : "+str(MPI.Wtime() - start))
        
        plt.plot(np.arange(0,256),Final_array_R,color='R',label = "Red Pixel")
        plt.plot(np.arange(0,256),Final_array_G,color='G', label = "Green Pixel")
        plt.plot(np.arange(0,256),Final_array_B,color='B' , label = "Blue Pixel")
        plt.ylabel('')
        plt.xlabel('')
        plt.legend()
        plt.show()
        
        
      
        
