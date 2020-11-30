# Tf-CoronaXray

## Model summary:
![summary](../main/readme/Summary.png) 
### 3 Convolution layers accompanied by max pooling layers with flatten to compress the parameters into a column vector containing all the contents of the larger vectors. Sigmoid used for the final binary classification with ReLU as hidden layer activation function. Mini-batches of 10 images were trained and validated every epoch (Total 15 epochs). The dataset was obtained from Kaggle (dated Feb. 2020).

![epochs](../main/readme/epoch.png)

## Model Test and validation data accuracy and loss visualised:
![Accplot](../main/readme/modelAcc.png) ![Lossplot](../main/readme/modelloss.png)

 
 
 
### P.s: The code in both main.py and main.ipynb is the same and i've just put them both here for flexibity of usage.
