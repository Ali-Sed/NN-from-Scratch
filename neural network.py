import neural_network_header as header
                
Hidden_size = 256
Batch_size = 5000
Preprocessing = False   ## Choose between FAlSE or TRUE
Type_of_preprocessing = 1 ## Choose between 1 or 2
Type_of_coding = 'One-hot' ## Choose between One-hot or Binary
Momentum = False            # Choose between False or True
    
Train_im, Train_lb, Test_im, Test_lb = header.Preparing_data ( Preprocessing , Type_of_preprocessing )
Train_lb , Test_lb , Out_size = header.Coding ( Train_lb, Test_lb,  Type_of_coding )
header.Mini_batch_grad_descent (  Out_size , Hidden_size, Batch_size , Test_im, Test_lb,Train_im , Train_lb , 784 , Momentum )











