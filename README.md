## Transformer-BiLSTM Based Lithium-ion Battery Remaining Useful Life (RUL) Prediction

### 1. Project Introduction

This project aims to utilize the NASA lithium-ion battery dataset to construct a deep learning model based on Transformer-BiLSTM for accurately predicting the remaining useful life (RUL) of lithium-ion batteries. With the widespread application of lithium-ion batteries in various fields, accurately predicting their remaining service life is of great significance for optimizing battery management systems, extending battery life, reducing maintenance costs, and improving system safety.

### 2. Directory Structure

```
.
├── dataset                 		        
│   ├── B0005.mat   						        
│	├── B0006.mat   						       
│	├── B0007.mat   						         
│	└── B0018.mat   						         
├── model                  					    
│   ├── best_model_LSTM.pt  				   
│   ├── best_model_Transformer.pt       
│   └── best_model_TransformerBiLSTM.pt 
├── picture                 				   
├── data_process.py							        
├── data_trans.py							        
├── model_compare.ipynb						     
├── model_test.py							         
├── model_train.py							       
├── Transformer_BiLSTM.py					      
├── requirements.txt        				  
└── README.md               				    
```

### 3. Environment Configuration

This project is developed using Python 3.11 and primarily depends on the following libraries:

- Python 3.11
- Pytorch 2.5.0
- CUDA 11.8 + cuDNN 8
- numpy==1.26.3
- pandas==2.2.3
- matplotlib==3.10.1
- scikit-learn==1.6.1

You can install the project dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 4. Dataset

This project uses the NASA Battery Dataset, which covers battery performance data under different operating conditions. The dataset includes experimental data of multiple lithium-ion batteries under different operating modes (charging, discharging, and impedance measurement), stored in the .mat file format.

Dataset download link: [NASA Battery Dataset](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)

Visualize the data by plotting the capacity degradation curves of four battery groups (B0005, B0006, B0007, B0018) at 24℃ to observe the variation trend of battery capacity with cycle counts.

<img src="picture\capacity.png" alt="capacity" style="zoom: 50%;" />

### 5. Model Definition: Transformer-BiLSTM

The `Transformer_BiLSTM.py` file contains the definition of the Transformer-BiLSTM model, which combines the global dependency capture capability of Transformer with the bidirectional feature extraction ability of BiLSTM.

<img src="picture\model.png" alt="model" style="zoom:60%;" />

### 6. Model Training

Training Configuration:

- **Device**: Training is performed on a GPU (NVIDIA GeForce RTX 4090 ).
- **Loss function**: Mean Squared Error (MSE) loss is used as the optimization objective.
- **Optimizer**: The Adam optimizer is used with a learning rate of 0.0001 and momentum parameters of (0.9, 0.999).
- **Learning rate scheduling**: The StepLR learning rate scheduler is used, reducing the learning rate to 0.9 times its original value every 10 epochs.
- **Training epochs**: A total of 50 epochs are trained.
- **Batch size**: Set to 32.

### 7. Model Testing

Load the trained model `best_model_TransformerBiLSTM.pt` , plot the prediction curves of the model on the training, validation, and test sets, and compare them with the true values. Evaluate the model's prediction performance on the training, validation, and test sets by calculating error metrics (MSE, RMSE, MAE, R²).

The prediction curves of the model on the training, validation, and test sets are shown in the following figures:

<img src="picture\bilstm-transformer-test1.png" alt="bilstm-transformer-test1" style="zoom:45%;" />

<img src="picture\bilstm-transformer-test2.png" alt="bilstm-transformer-test2" style="zoom:40%;" />

<img src="picture\bilstm-transformer-test3.png" alt="bilstm-transformer-test3" style="zoom:40%;" />

The error metrics of the model on the training, validation, and test sets are as follows:

| Dataset/Metric | R²     | MSE      | RMSE   | MAE    |
| -------------- | ------ | -------- | ------ | ------ |
| Training set   | 0.9233 | 0.0035   | 0.0589 | 0.0323 |
| Validation set | 0.9701 | 0.000724 | 0.0269 | 0.0221 |
| Test set       | 0.9670 | 0.000755 | 0.0275 | 0.0200 |

### 8. Model Performance Comparison

To further explore the superiority of the Transformer-BiLSTM model in battery capacity degradation prediction tasks, we constructed Transformer and LSTM models and compared the performance of the three models.

The prediction curves of the three models on the test set (B0018) are shown in the following figure:

<img src="picture\compare1.png" alt="compare1" style="zoom:40%;" />

The error metrics of the three models on the test set (B0018) are as follows:

| Model              | R²     | MSE      | RMSE   | MAE    |
| ------------------ | ------ | -------- | ------ | ------ |
| LSTM               | Low    | High     | High   | High   |
| Transformer        | High   | Medium   | Medium | Medium |
| Transformer-BiLSTM | 0.9670 | 0.000755 | 0.0275 | 0.0200 |

From the model comparison results, it is evident that the Transformer-BiLSTM model excels in this task. By combining the advantages of BiLSTM and Transformer, it can effectively capture long-term dependencies in time series data and comprehensively model sequence dependencies through self-attention mechanisms. This enables efficient parallel computation, making it the optimal choice for battery capacity degradation prediction tasks.

### References

[1] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[J]. Advances in neural information processing systems, 2017, 30.

[2] A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data Set, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

[3] LI B, HAN X, ZHANG W, et al. Review of the remaining useful life prediction methods for lithium ion batteries[J]. Energy Storage Science and Technology, 2024, 13(4): 1266.

[4] Zhang Y, Xiong R, He H, et al. Long short-term memory recurrent neural network for remaining useful life prediction of lithium-ion batteries[J]. IEEE Transactions on Vehicular Technology, 2018, 67(7): 5695-5705.

[5] Wu Xiaodan, Fan Bo, Wang Jianxiang. Lithium Battery Life Prediction Based on VMD-TCN-Attention[J]. Power Technology, 2023, 47(10): 1319-1325.

[6] Niu Qunfeng, Yuan Qiang, Wang Li, et al. Lithium-ion Battery Remaining Life Prediction Based on CEEMDAN-RVM-LSTM Model[J]. Power Technology, 2023, 47(10): 1313-1318. DOI:10.3969/j.issn.1002-087X.2023.10.017.

[7] Chen Hongxia, Ding Guorong, Chen Guici. Lithium-ion Battery Health State Estimation Based on Variational Mode Decomposition and Long Short-term Memory Network [J/OL]. Journal of Power Sources, 1-13 [2024-01-18]. CHEN HX, DING GR, CHEN GC, et al.

[8] Chen D, Zhou X. AttMoE: Attention with Mixture of Experts for remaining useful life prediction of lithium-ion batteries[J]. Journal of Energy Storage, 2024, 84: 110780.
