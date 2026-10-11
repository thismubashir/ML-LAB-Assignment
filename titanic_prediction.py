import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

data = pd.read_csv("sample-data.csv")
print(data.head())
print("Rows:", len(data))

print(data.isnull().sum())
print(data['Survived'].value_counts())
data = data.dropna().copy()

features = ['PClass', 'Gender', 'Sibling', 'Embarked']
target = 'Survived'
data = data[features + [target]]

encoded = data.copy()
encoders = {}

for column in data.columns:
    encoder = LabelEncoder()
    encoded[column] = encoder.fit_transform(data[column])
    encoders[column] = encoder

encoded.to_csv("sample-data-encoded.csv", index=False)
output_data = data.copy()
output_data[target] = encoded[target]
output_data.to_csv("sample-data-encoded-output.csv", index=False)
output_data = data.copy()
output_data[target] = encoded[target]
output_data.to_csv("sample-data-encoded-output.csv", index=False)
print(encoded.head())

training, testing = train_test_split(
    encoded, test_size=0.2, random_state=0, shuffle=False
)

training.to_csv("training-data-encoded.csv", index=False)
testing.to_csv("testing-data-encoded.csv", index=False)

X_train = training[features]
y_train = training[target]
X_test = testing[features]
y_test = testing[target]

model = SVC(gamma="auto", random_state=0)
model.fit(X_train, y_train)

with open('svc_trained_model.pkl', "wb") as file:
    pickle.dump(model, file)

with open('svc_trained_model.pkl', "rb") as file:
    model = pickle.load(file)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy, 2))
print(classification_report(y_test, predictions, target_names=encoders[target].classes_))

results = testing.copy()
results["Predictions"] = predictions
results.to_csv("model-predictions.csv", index=False)

ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=encoders[target].classes_, cmap="Blues"
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

example = pd.DataFrame([['First', 'Male', 'Zero', 'Queenstown']], columns=features)
example_encoded = example.copy()

for column in features:
    example_encoded[column] = encoders[column].transform(example[column])

prediction = model.predict(example_encoded)
print("Prediction:", encoders[target].inverse_transform(prediction)[0])

improvements = ['Collect a larger Titanic dataset.', 'Compare SVC with other classifiers.']

for item in improvements:
    print(item)
