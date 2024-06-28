Our algorithm achieves Top3.

### Reproduction process

Run train. sh and test. sh in sequence. If the results do not match the results during the competition, please manually run train. py and test. py in sequence.
Train is responsible for training the model and saving it to the user_data folder;
The test is responsible for using the model trained in the train and saving the prediction results CSV file to the prediction_result folder.
### Algorithm introduction
Considering the characteristics of structured data and the need to suppress overfitting, the catboost model was used for this prediction.
In train and test, merge train.csv and test.csv to form an overall dataframe, which facilitates the construction of various features and the use of corresponding rows during training and prediction.
Since PM2.5 and other data are not included in the test set, it is also necessary to make predictions on these data.
The data from each monitoring station is correlated, so the data from 17 stations are input into the model for training and prediction.

##### Data processing and feature engineering
Smooth continuous data with log1p to reduce the impact of outliers and make the data more in line with Gaussian distribution;
During the model debugging process, it was found that the Toluene column poses certain risks to prediction, so it was discarded;
Considering the severe missing data from a distance and its limited significance for current predictions, the data from before September 2019 was discarded;
Through period detection, it was found that the data has strong periodicity for month and hour, and time features for month and hour were extracted from the date;
Move the columns in shift_features back 24 and 48 hours as basic features to reflect the historical information of the time series;
On the basis of these time shift features, long short-term mean features and max statistical features within sliding windows were constructed to reflect the trend of time series changes;
The data between monitoring stations has a certain correlation, and the average value of data from some stations at the same time is taken as the feature;
Taking the average of data from the same monitoring station and hour in historical data as a feature, in order to increase the amount of data participating in the average and reduce the impact of extreme values, data from before September 2019 was used in this feature.
##### Model construction
In the training model, due to the large number of columns to be predicted, not only targets but also features such as PM2.5 need to be predicted for future prediction. As the cost of model tuning is relatively high, automatic early stopping is mainly used to ensure model performance. However, this method loses data from the validation set. Therefore, the model is constructed twice, with the first iteration obtaining the number of early stopping iterations, and the second time combining the first training set and validation set into the training set, using the first iteration.

##### Prediction

##### During the prediction process, predict the data for the next 24 hours for a total of 30 times, and use the results of this prediction for the next prediction.
This method of training the model only once significantly reduces time costs compared to using predicted data to train the model multiple times, and to some extent prevents error accumulation, resulting in better prediction results.