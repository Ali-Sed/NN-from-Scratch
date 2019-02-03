from mlxtend.data import loadlocal_mnist
import numpy as np
import matplotlib.pyplot as plt

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
    
    Title = str('Batch_size = '+ str(Batch_size)+' Hid_Layer_size = ' + str(Hidden_size)+ ' Active_Func = ' + Active_func_name )
    plt.title (Title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    
    x_axis = np.arange(1, epoch , 1)
    plt.plot(x_axis , List)
    plt.show()

def Preparing_data ( Preprocessing = False , Type_of_preprocessing = 1  ):
    
    Train_im , Train_lb = loadlocal_mnist(images_path ='train-images.idx3-ubyte',
                                          labels_path = 'train-labels.idx1-ubyte')

    Test_im , Test_lb = loadlocal_mnist(images_path = 't10k-images.idx3-ubyte',
                                        labels_path = 't10k-labels.idx1-ubyte')

    if ( Preprocessing == True ):
        
        if ( Type_of_preprocessing == 1):
            for i in range ( len ( Train_im) ):
                mean = np.mean ( Train_im[i] )
                std  = np.std  ( Train_im[i] )
                Train_im[i] = (Train_im[i] - mean) / (std+0.2)
                
            for i in range ( len ( Test_im) ):
                mean = np.mean ( Test_im[i] )
                std  = np.std  ( Test_im[i] )
                Test_im[i] = (Test_im[i] - mean) / (std+0.2)
                
        elif ( Type_of_preprocessing == 2) :
            
            for i in range ( len ( Train_im) ):
                x_max = np.max ( Train_im[i] )
                x_min = np.min ( Train_im[i] )
                Train_im[i] = (Train_im[i] - x_min)/ (x_max - x_min + 0.1)
                
            for i in range ( len ( Test_im) ):
                x_max = np.max ( Test_im[i] )
                x_min = np.min ( Test_im[i] )
                Test_im[i] = (Test_im[i] - x_min)/ (x_max - x_min + 0.1)
                
    return Train_im, Train_lb, Test_im, Test_lb

def Coding ( Train_lb, Test_lb , Type_of_coding = 'One-hot'  ):
    
    if ( Type_of_coding == 'One-hot'):
        Out_size = 10
        
        Zero = np.zeros((10,60000))
        for i in range (len(Train_lb)):
            Zero[Train_lb[i],i] = 1
        Train_lb = Zero
        
        Zero = np.zeros((10,10000))
        for i in range (len(Test_lb)):
            Zero[Test_lb[i],i] = 1
        Test_lb = Zero
        
    elif ( Type_of_coding == 'Binary'):
        
        Out_size = 4
        Zero = np.zeros((4,60000))
        for i in range (len(Train_lb)):
            Bin = '{0:04b}'.format(Train_lb[i])
            for j in range (4):
                Zero[j,i] = Bin[j]
        Train_lb = Zero
        
        Zero = np.zeros((4,10000))
        for i in range (len(Test_lb)):
            
            Bin = '{0:04b}'.format(Test_lb[i])
            for j in range (4):
                Zero[j,i] = Bin[j]
        Test_lb = Zero
        
    return Train_lb , Test_lb , Out_size



def Testing  ( Test_im, Test_lb , W_2,W_1,Bias_1, Bias_2 , Active_func_name):
    
    Z_2 = np.dot ( W_1 , Test_im.T  ) +(Bias_1)
    A_2 = Active_func (Active_func_name , Z_2)
    Z_3 = np.dot ( W_2, A_2) + Bias_2 
    Y   = Active_func (Active_func_name, Z_3)
    accuracy = 0
    index_one =  np.argmax(Test_lb , axis = 0)
    
    for j in range (10000):
             
        for k in range ( Out_size ):
            if Y[k][j] == max (Y[:,j]) :
                index_max = k
            
        if index_max == index_one[j] :
            accuracy +=1
       
    accuracy /= 10000 
    return accuracy 
    

        
        
   
def Mini_batch_grad_descent ( Out_size , Hidden_size, Batch_size  ,Test_im, Test_lb ,Train_im , Train_lb , Feature_size, Momentum = False  ):
    
    epoch = 1

                        
    W_1 = np.random.rand (Hidden_size  , Feature_size )*0.001
    W_2 = np.random.rand (Out_size , Hidden_size)*0.001
    Bias_1 = np.random.rand ( Hidden_size, 1)*0
    Bias_2 = np.random.rand ( Out_size, 1)*0
    V_Grad_W_1 , V_Grad_W_2 , V_Grad_Bias_1, V_Grad_Bias_2 = 0 , 0 ,0 ,0
    for Active_func_name in [ 'Sigmoid' ,  'Tanh', 'Line']:
        for Batch_size in range ( 100, 501 ,200):
            for Hidden_size in range( 150 ,151,100):


                if Active_func_name == 'Sigmoid' :
                    Eta = 0.01
                elif Active_func_name == 'Tanh':
                    Eta = 0.01
                else:
                    Eta = 0.0001
                    
                accuracy = 0
                Cost_list = []
                Error_list =[]
                
                while (accuracy<0.92 and epoch <35):
                    for i in range( int (len(Train_im) / Batch_size)):
                        accuracy = 0

                        Z_2 = np.dot ( W_1 , Train_im[i*Batch_size: (i+1)*(Batch_size)].T  ) +(Bias_1)
                        A_2 = Active_func (Active_func_name , Z_2)
                        Z_3 = np.dot ( W_2, A_2) + Bias_2 
                        Y   = Active_func (Active_func_name, Z_3)
                                            
                        Grad_Z_3 =  (Y - Train_lb[:,i*Batch_size: (i+1)*(Batch_size)] ) * Prime_active_func(Active_func_name , Y)
                        Grad_Z_2 = np.dot( W_2.T , Grad_Z_3) * Prime_active_func(Active_func_name , A_2)
                        Grad_W_2 = 1/ Batch_size * np.dot( Grad_Z_3 , A_2.T )
                        Grad_W_1 = 1/ Batch_size *np.dot  ( Grad_Z_2, Train_im[i*Batch_size: (i+1)*(Batch_size) , 0:Feature_size]   )  
                        Grad_Bias_1 = 1/ Batch_size * np.sum(Grad_Z_2 , axis = 1 , keepdims = True)
                        Grad_Bias_2 = 1/ Batch_size * np.sum(Grad_Z_3 , axis = 1 , keepdims = True)

                        if ( Momentum == True ):
                            V_Grad_W_1    = 0.9 * V_Grad_W_1 + 0.1 * Grad_W_1
                            V_Grad_W_2    = 0.9 * V_Grad_W_2 + 0.1 * Grad_W_2
                            V_Grad_Bias_1 = 0.9 * V_Grad_Bias_1 + 0.1 * Grad_Bias_1
                            V_Grad_Bias_2 = 0.9 * V_Grad_Bias_2 + 0.1 * Grad_Bias_2

                            W_2    -= Eta * V_Grad_W_2 
                            W_1    -= Eta * V_Grad_W_1
                            Bias_1 -= Eta * V_Grad_Bias_1
                            Bias_2 -= Eta * V_Grad_Bias_2

                            
                            

                        else:    
                            W_2    -= Eta * Grad_W_2 
                            W_1    -= Eta * Grad_W_1
                            Bias_1 -= Eta * Grad_Bias_1
                            Bias_2 -= Eta * Grad_Bias_2

                        index_one =  np.argmax(Train_lb[:,i*Batch_size: (i+1)*(Batch_size)], axis = 0)

                        for j in range (Batch_size):
                            for k in range ( Out_size ):
                                if Y[k][j] == max (Y[:,j]) :
                                    index_max = k
                                
                            if index_max == index_one[j] :
                                accuracy +=1
                    
                    accuracy /= Batch_size
                    print ('accuracy = ' , accuracy)           
                    Cost = Cost_func ( Y , Train_lb[:, i*Batch_size: (i+1)*(Batch_size)])                    
                    Cost_list.append(Cost)
                    Error_list.append(1 - accuracy)
                    print ('iter = ' ,epoch)
                    epoch +=1

                Plot ( epoch,Error_list, Batch_size, Hidden_size, Active_func_name, 'Error' )
                Plot ( epoch, Cost_list, Batch_size, Hidden_size, Active_func_name, ' Cost' )
                
                Test_accuracy = Testing ( Test_im, Test_lb , W_2,W_1,Bias_1, Bias_2 , Active_func_name)
                print ('Test Accuracy = ' , Test_accuracy)
                
                epoch = 1
                W_1 = np.random.rand (Hidden_size  , Feature_size )*0.001
                W_2 = np.random.rand (Out_size , Hidden_size)*0.001
                Bias_1 = np.random.rand ( Hidden_size, 1)*0
                Bias_2 = np.random.rand ( Out_size, 1)*0

                V_Grad_W_1 , V_Grad_W_2 , V_Grad_Bias_1, V_Grad_Bias_2 = 0 , 0 ,0 ,0
