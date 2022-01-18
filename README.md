# Machine_Learning-Project 
Hello all,
</br>
Nice to see you exploring my machine learning project which I made during my 5th semester.
</br>
The project's main aim is to the **predict the health of crops** of various types given its exposure to various factors such as soil type, insect counts,season, pesticides etc
</br>
We use classification models in </br>
## Machine Learning such as: </br>
*Logistic Regression* </br>
*K-Nearest-Neighbors* </br>
*Random Forest Classifier*</br>
*Decision Tree Classifier* to classify our test instances into 3 various categories namely: <br>
## 1) No Damage</br>
## 2) Damage due to use of pesticides </br>
## 3) Damage due to other factors </br>
> After initial look up of the dataset we realised that there were missing values in one of the columns and hence we went about using fillna method to replace all the missing values
by their mean value</br>

**Correlation matrix**</br>
## Correlation Matrix
<img src="Crop_Health_prediction/images/correlation.png" width="400" height="400"></br>
>From the correlation matrix we found out that there were a few irrelevant features in the dataset and few were highly positively correlated and hence
 would be useful in classification
</br>1.Estimated_Insects_count,Pesticide_use_category and Number_weeks_used are positively correlated with Crop damage.</br>
2.Number_weeks_used  is positively correlated with Estimated_Insects_count and Pesticide_use_category.</br>
