import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn import pipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv(r"/Users/daniel_kristensen/DAKI/opgaver/P1_kode/data.csv")

# Define features and target variable
features = ['Cholesterol', 'RestingBP', 'Age', 'FastingBS', 'MaxHR', 'Sex_M', 
            'ExerciseAngina_Y', 'RestingECG_ST', 'ChestPainType_TA', 
            'ChestPainType_ATA', 'ChestPainType_NAP', 'RestingECG_Normal']
target = 'HeartDisease'

X = df[features]
y = df[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13, stratify=y)


# Define scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}


# -----------


#Ensemble
log_reg =  Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=500, solver="liblinear", random_state=13))])

svm = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(probability=True, kernel="rbf", random_state=13))
])

rf = RandomForestClassifier(n_estimators=200,max_depth=20, random_state=13)

# Create voting classifier
voting_model = VotingClassifier(
    estimators=[
        ('log_reg', log_reg),
        ('svm', svm),
        ('rf', rf)
    ],
    voting='soft'  
)

cv_results = cross_validate(voting_model, X, y, cv=20, scoring=scoring)

#Reaults
print("Stack:")
print("Mean cross-validation F1 score:", cv_results['test_f1'].mean())
print("Mean cross-validation accuracy:", cv_results['test_accuracy'].mean())
print("Mean cross-validation precision:", cv_results['test_precision'].mean())
print("Mean cross-validation recall:", cv_results['test_recall'].mean())


# ----------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Custom threshold value
threshold = 0.376

def evaluate_threshold(voting_model, X_train, y_train, X_test, y_test, threshold):
    voting_model.fit(X_train, y_train)

    y_pred_proba = voting_model.predict_proba(X_test)[:, 1]

    y_pred_custom = (y_pred_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    accuracy = accuracy_score(y_test, y_pred_custom)

    return y_pred_custom, precision, recall, f1, accuracy

# Custom threshold
y_pred_custom, precision, recall, f1, accuracy = evaluate_threshold(
    voting_model, X_train, y_train, X_test, y_test, threshold
)

# Print results
print(f"Custom Threshold Metrics at Threshold {threshold}:")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")