# CODSOFT-5


Importing Libraries:


pandas for data manipulation and analysis.
numpy for numerical operations.
matplotlib.pyplot and seaborn for data visualization.
sklearn for machine learning tasks, including data splitting, preprocessing, and model evaluation.
imblearn for dealing with imbalanced datasets.
collections for counting class occurrences.
Loading the Dataset:

The code loads a credit card fraud dataset from a CSV file using pd.read_csv() and stores it in a DataFrame called data.
Exploring the Dataset:

It prints the first few rows of the dataset using data.head() to get a glimpse of the data.
It checks the class balance by counting the occurrences of each class (fraudulent and non-fraudulent) using value_counts() and stores the result in class_counts.
It visualizes the class distribution using a countplot from seaborn to see if there's a class imbalance issue.
Data Preprocessing:

It separates the dataset into input features (X) and the target variable (y) where X contains all columns except the 'Class' column, and y contains the 'Class' column.
It standardizes the 'Amount' feature using StandardScaler to scale it to have a mean of 0 and a standard deviation of 1.
Train-Test Split:

The dataset is split into training and testing sets using train_test_split() from sklearn. It allocates 80% of the data for training (X_train and y_train) and 20% for testing (X_test and y_test).
Handling Class Imbalance:

It addresses the class imbalance issue by oversampling the minority class using RandomOverSampler from imblearn. This is done to ensure that the model is not biased towards the majority class. The oversampling strategy is set to a ratio of 0.5, which means the minority class will be increased to be 50% of the majority class.
Training the Random Forest Classifier:

A Random Forest Classifier is initialized with a specified random state.
The classifier is trained on the oversampled training data (X_train_resampled and y_train_resampled) using clf.fit().
Model Evaluation:

The trained model is used to make predictions on the test data (X_test) using clf.predict().
Various metrics are calculated and printed, including accuracy, confusion matrix, and classification report, to assess the model's performance.
Visualization:

The code visualizes the confusion matrix using seaborn's heatmap.
It also displays the feature importance using bar plots to show which features were most influential in the model's decision-making process.
Sorting Feature Importance:

Feature importances obtained from the Random Forest model are sorted in descending order.
A bar chart is created to visualize the sorted feature importances, helping to identify the most important features.



