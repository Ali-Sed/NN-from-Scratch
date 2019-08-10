from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pyplot as plt
import neural_network_header as header


Out_size = 784

def Cost_func ( Y , Target ):
    Cost = sum ( sum ( ( Y - Target) **2 ) ) / len(Y)
    return Cost

def Sigmoid (X):
    return  1 / (1 + np.exp(-X))

def Active_func( Active_func_name , X ):
    if ( Active_func_name == 'Sigmoid' ):
        return Sigmoid(X)
    elif ( Active_func_name == 'Tanh' ):
        return np.tanh(X)
    elif (Active_func_name == 'Line'):
        return X

def Prime_active_func( Active_func_name , X ):
    if ( Active_func_name == 'Sigmoid' ):
        return Sigmoid(X) * (1 - Sigmoid(X))
    elif ( Active_func_name == 'Tanh' ):
        return 1 - np.tanh(X)**2
    elif (Active_func_name == 'Line'):
        return 1

def Plot ( epoch, List , Batch_size, Hidden_size, Active_func_name , ylabel):
    
    Title = str('Batch_size = '+ str(Batch_size)+' Hid_Layer_size = ' + str(Hidden_size)+ ' Auto Encoder' )
    plt.title (Title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    
    x_axis = np.arange(1, epoch , 1)
    plt.plot(x_axis , List)
    plt.show()


def Mini_batch_grad_descent (Test_im, Test_lb ,Train_im , Train_lb , Momentum = False ):
    
    global Out_size
    epoch = 1
    Active_func_name = 'Sigmoid'
    Batch_size = 6000
    Hidden_size = 100

    Eta = 0.01
    accuracy = 0
    Cost = 100
    Cost_list = []
    Error_list =[]

            
    W_1 = np.random.rand (Hidden_size  , 784 )*0.0001
    W_2 = np.random.rand (Out_size , Hidden_size)*0.0001
    Bias_1 = np.random.rand ( Hidden_size, 1)*0
    Bias_2 = np.random.rand ( Out_size, 1)*0
    while (epoch <15):
        
        for i in range( int (len(Train_im) / Batch_size)):
            accuracy = 0

            Z_2 = np.dot ( W_1 , Train_im[i*Batch_size: (i+1)*(Batch_size)].T  ) +(Bias_1)
            A_2 = Active_func (Active_func_name , Z_2)
            Z_3 = np.dot ( W_2, A_2) + Bias_2 
            Y   = Active_func (Active_func_name, Z_3)
                                
            Grad_Z_3 =  (Y - Train_lb[i*Batch_size: (i+1)*(Batch_size)].T ) * Prime_active_func(Active_func_name , Y)
            Grad_Z_2 = np.dot( W_2.T , Grad_Z_3) * Prime_active_func(Active_func_name , A_2)
            Grad_W_2 = 1/ Batch_size * np.dot( Grad_Z_3 , A_2.T )
            Grad_W_1 = 1/ Batch_size *np.dot  ( Grad_Z_2, Train_im[i*Batch_size: (i+1)*(Batch_size) , 0:784]   )  
            Grad_Bias_1 = 1/ Batch_size * np.sum(Grad_Z_2 , axis = 1 , keepdims = True)
            Grad_Bias_2 = 1/ Batch_size * np.sum(Grad_Z_3 , axis = 1 , keepdims = True)


            W_2    -= Eta * Grad_W_2 
            W_1    -= Eta * Grad_W_1
            Bias_1 -= Eta * Grad_Bias_1
            Bias_2 -= Eta * Grad_Bias_2

        Cost = Cost_func ( Y , Train_lb[ i*Batch_size: (i+1)*(Batch_size)].T)                    
        Cost_list.append(Cost)
        print ('iter = ' ,epoch)
        print ('cost= ', Cost )
        epoch +=1

    Plot ( epoch, Cost_list, Batch_size, Hidden_size, Active_func_name, ' Cost' )
    return W_1 , Bias_1



Train_im , Train_lb = loadlocal_mnist(images_path ='train-images.idx3-ubyte',
                                      labels_path = 'train-labels.idx1-ubyte')

Test_im , Test_lb = loadlocal_mnist(images_path = 't10k-images.idx3-ubyte',
                                    labels_path = 't10k-labels.idx1-ubyte')

W_1 , Bias_1 = Mini_batch_grad_descent ( Test_im, Test_im,Train_im , Train_im)
Hidden_size = 50
Batch_size  = 1000
Momentum = False
Out_size = 10

Zero = np.zeros((10,60000))
for i in range (len(Train_lb)):
    Zero[Train_lb[i],i] = 1
Train_lb = Zero

Zero = np.zeros((10,10000))
for i in range (len(Test_lb)):
    Zero[Test_lb[i],i] = 1
Test_lb = Zero

Z_2 = np.dot ( W_1 , Train_im.T  ) +(Bias_1)
Train_im = header.Active_func ('Sigmoid' , Z_2)
header.Mini_batch_grad_descent (  Out_size , Hidden_size, Batch_size , Test_im, Test_lb,Train_im.T , Train_lb ,100,Momentum)






