# Email_Spam_Detector 

The code contains a comparison of two machine learning models, KNN and Decision trees. The models were trained using the Spambase dataset and evaluated based on:

Accuracy
F1 Score
Precision
Recall
Confusion Matrix
The Spambase dataset and it's documentation can be found in the following URL:

https://archive.ics.uci.edu/ml/datasets/Spambase

The model with the best accuracy score is selected and imported and used as .pkl file.

The imported model is then used to detect spam in text input by the user. The user can input raw text from an email and the text thus added is converted to numerical format using the extract_features_from_email(raw_text)
The extract_features_from_email(raw_text) function convert the raw text data added by the user to a numerical format that can be used with model. 
