# Tf-CoronaXray

## Model summary:
![summary](../main/readme/Summary.png) 
<p> 3 Convolution layers accompanied by max pooling layers with flatten to compress the parameters into a column vector containing all the contents of the larger vectors. Sigmoid used for the final binary classification with ReLU as hidden layer activation function. Mini-batches of 10 images were trained and validated every epoch (Total 15 epochs). The dataset was obtained from Kaggle (dated Feb. 2020).</p>

![epochs](../main/readme/epoch.png)
### The very unusual looking high validation accuracy and fluctuating training accuracy is currently due to the small size of the dataset in comparison to my model, data augmentation could have been performed but i decided to wait on larger more clearer datasets to apply Transfer learning later on.


![Accplot](../main/readme/modelAcc.png) ![Lossplot](../main/readme/modelloss.png)

 
 
 
### P.s: The code in both main.py and main.ipynb is the same and i've just put them both here for flexibity of usage.
