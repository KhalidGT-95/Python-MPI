import cv2
import numpy as np
from mpi4py import MPI
import pylab as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Histogram_array = np.zeros(256,int) # Used by each process to store the partial values 
Final_array = np.zeros(256,int)     # Used for collecting partial values from each process and merging them together in a final array for generating histogram

def GrayScale(image):
    global Histogram_array
    global Final_array
    
    # len(image) -> returns the number of rows
    # len(image[0]) -> returns the number of columns
    # intensity of the pixel lies between 0-255
    
    for i in range(rank,len(image),size): 
        for j in range(len(image[0])):
            value = image[i][j]      # Get the intensity of the pixel 
            
            Histogram_array[value] += 1     # Increment the index of that intensity by 1
    
    comm.Reduce(Histogram_array,Final_array,root=0,op=MPI.SUM)    # Every process sends its result to the root process. The MPI.SUM reduce operation is used which sums all the partial values
    
if __name__=="__main__":

    gray_img = cv2.imread('2048.jpg',cv2.IMREAD_GRAYSCALE)  # read the image in grayscale mode
    
    start = MPI.Wtime()     # start the timer
        
    GrayScale(gray_img) 
    
    if rank == 0:
        print("\nTotal pixels present in the picture are : " + str(sum(Final_array)) + ". \nThis serves as a proof that we have processed each pixel of the image since the total sum comes out exactly same as the number of pixels")
        
        print("\nThe time taken is : " + str(MPI.Wtime() - start))
        plt.plot(np.arange(0,256),Final_array,color='Gray',label = "GrayScale")
        
        plt.ylabel('')
        plt.xlabel('')
        plt.legend()
        plt.show()
        
        
