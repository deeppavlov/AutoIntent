from autointent.modules.scoring._cnn.cnn import CNNScorer

# Sample data
utterances = [
    "I love programming",
    "I hate bugs",
    "Python is awesome",
    "Debugging is frustrating",
    "Machine learning is fun",
    "I dislike errors",
]
print(utterances)
labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Initialize the scorer
scorer = CNNScorer()
# Train the model
print('before fit')
scorer.fit(utterances, labels)

# Test set
test_utterances = [
    "I enjoy coding",
    "I find bugs annoying",
    "AI is fascinating",
    "Errors are frustrating",
]

# Predict probabilities
probabilities = scorer.predict(test_utterances)
print("Predicted Probabilities:")
print(probabilities)

# Convert probabilities to predicted labels
predicted_labels = (probabilities > 0.5).astype(int)  # For binary classification
print("Predicted Labels:")
print(predicted_labels)

# Expected labels for the test set
expected_labels = [1, 0, 1, 0]

# Compare predicted and expected labels
for i, (pred, exp) in enumerate(zip(predicted_labels, expected_labels)):
    print(f"Test Utterance {i+1}: {test_utterances[i]}")
    print(f"Predicted: {pred}, Expected: {exp}")