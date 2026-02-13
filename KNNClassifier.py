import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data_from_csv(file_name):

    """Load dataset from a CSV file and separate features and class labels. From Class Example."""

    data = np.genfromtxt(file_name, delimiter=',', names=True, filling_values=0, dtype=None, \
ndmin=1)
    # Extract all except the last column name as feature names
    feature_names = list(data.dtype.names[:-1])
    # Extract unique values from last column as class names
    class_names = np.unique(data[data.dtype.names[-1]]).tolist()
    # Extract all except the last column as features
    features = data[feature_names].tolist()
    # Extract last column as classes
    np_classes = list(data[data.dtype.names[-1]])
    classes = [str(item) for item in np_classes]

    return features,classes,feature_names,class_names

def clean(df):

    """Keep only numeric columns in a dataset."""

    # Convert to dataframe
    df = pd.DataFrame(df)
    # Only include numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    return numeric_df

def get_predictions(K,features_train,classes_train,features_test):

    """Train a K-Nearest Neighbors classifier and generate predictions. From Class Example."""

    # Initialize the K-NN Classifier
    knn = KNeighborsClassifier(n_neighbors=K)
    # Train the model
    knn.fit(features_train, classes_train)
    # Predict for the test data
    predictions = knn.predict(features_test)

    return knn,predictions

def find_best_k(max_k, features_train, classes_train, features_test, classes_test):

    """Determine the best K value from 1 to max_k based on accuracy."""
    
    # Initialization
    accuracies = []
    best_k = 1
    best_accuracy = 0

    # Loop from 1 to max_k
    for k in range(1, max_k + 1):
        knn, predictions = get_predictions(k, features_train, classes_train, features_test)
        
        acc = accuracy_score(classes_test, predictions)
        accuracies.append([k, acc])
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k

    # Save Accuracy.csv
    accuracy_df = pd.DataFrame(accuracies, columns=["K", "Accuracy"])
    accuracy_df.to_csv("Accuracy.csv", index=False)

    print(f'From 1 to {max_k}, The best K is: {best_k}')
    print(f'The best accuracy is: {best_accuracy}')

    return best_k, best_accuracy, accuracy_df

def generate_predictions(knn, features_test, classes_test, feature_names, testing_csv_path):
 
    """Generate predictions, compute confidence scores, save results to Testing.csv. Modified from Class Example."""

    predictions = knn.predict(features_test)
    probability = knn.predict_proba(features_test)
    confidences = np.max(probability, axis=1)

    # Load original Testing.csv
    testing_df = pd.read_csv(testing_csv_path)

    # Add Prediction and Confidence columns
    testing_df["Prediction"] = predictions
    testing_df["Confidence"] = confidences

    # Overwrite original Testing.csv
    testing_df.to_csv(testing_csv_path, index=False)

if __name__ == "__main__":

    # Load training and testing data
    features_train,classes_train,feature_names,class_names = get_data_from_csv("Training.csv")
    features_test,classes_test,*_ = get_data_from_csv("Testing.csv")

    # Normalization
    scaler = StandardScaler()
    features_train = clean(features_train)
    features_test = clean(features_test)
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Find the best K value and accuracy
    best_k, best_accuracy, accuracy_df = find_best_k(
        100,
        features_train,
        classes_train,
        features_test,
        classes_test
    )

    # Train with the best K and predict for the test data
    knn, _ = get_predictions(
        best_k,
        features_train,
        classes_train,
        features_test
    )

    #  Generate predictions and save results to Testing.csv
    generate_predictions(
        knn,
        features_test,
        classes_test,
        feature_names,
        "/content/Testing.csv"
    )