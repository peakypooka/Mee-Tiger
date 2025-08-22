import pandas as pd
#from src import postgres
import xgboost as xgb
from sklearn.datasets import load_iris # Load a sample dataset
from sklearn.model_selection import train_test_split

#Setting up the CSV file
csv_files = "d1.csv"
print(f"Reading CSV file: {csv_files}", flush=True)
data = pd.read_csv(csv_files)
print("CSV file read successfully.", flush=True)

# Load the dataset
print("Loading dataset...", flush=True)


iris = load_iris()
X, y = iris.data, iris.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating DMatrix for XGBoost
print("Creating DMatrix for XGBoost...", flush=True)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
print("DMatrix created successfully.", flush=True)

#Configuring XGBoost parameters
params = {
    'objective': 'multi:softprob', # Multi-class classification
    'num_class': 3,
    'max_depth': 5,
    'eta': 0.1,
    'eval_metric': 'mlogloss' # mlogloss for multi-class classification, eval_metric can be changed based on the task
}
print("Configuring XGBoost parameters...", flush=True)

# Training the model
print("Training the XGBoost model...", flush=True)
num_round = 100
bst = xgb.train(params, dtrain, num_round) # Iterations can be adjusted based on the dataset size and complexity
print("Model trained successfully.", flush=True)

print("Predicting...", flush=True)
preds = bst.predict(dtrain)
print("First 10 predictions:", preds[:10])
