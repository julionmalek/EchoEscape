import joblib

# Load the model from the file
model = joblib.load('political_bias_model.pkl')

# Define the text you want to classify
texts = ["In all, seven hostages have been rescued by troops alive, and the bodies of 19 hostages have also been recovered, including three mistakenly killed by the military."]

# Use the model to predict the bias
predicted_labels = model.predict(texts)

# Print the results
print("Predicted Labels:", predicted_labels)
