import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
import numpy as np
import joblib

# Load and clean the dataset
df = pd.read_csv("combined_log.csv") # Load the combined log file
df = df[
    (df["health"] <= 200) & # filter outliers
    (df["prev_health_3s"] <= 200) & # filter outliers
    (df["primary_ammo"] <= 200) & # filter outliers
    (df["secondary_ammo"] <= 200) # filter outliers
].copy()
df["reload_label"] = (df["action_took"] == "reload").astype(int)

# Encode the previous actions (last 3 actions)
action_encoder = LabelEncoder()
df["action_took_encoded"] = action_encoder.fit_transform(df["action_took"].fillna("hold"))

# Balance the dataset by upsampling the minority class (reload actions)
reload_df = df[df["reload_label"] == 1] # Filter reload actions
no_reload_df = df[df["reload_label"] == 0] # Filter non-reload actions
reload_df_upsampled = resample(reload_df, replace=True, n_samples=len(no_reload_df), random_state=42) # Upsample
df = pd.concat([no_reload_df, reload_df_upsampled]).sample(frac=1, random_state=42) # Combine and shuffle

# Function to create sequences of actions and features for training
def create_sequences(dataframe, seq_len=3): 
    sequences = [] # List to store sequences
    labels = [] # List to store labels
    for i in range(seq_len, len(dataframe)):
        # Extract the sequence of actions 
        actions_seq = dataframe["action_took_encoded"].values[i - seq_len:i]
        # Extract the features for the current row
        features = dataframe.iloc[i][[
            "health", "primary_ammo", "primary_in_mag",
            "secondary_ammo", "secondary_in_mag",
            "prev_health_3s", "enemy_shown"
        ]].values.astype(float)
        # Combine actions and features into a single sequence
        full_seq = np.concatenate((actions_seq, features))
        sequences.append(full_seq)
        labels.append(dataframe.iloc[i]["reload_label"]) # Append the label for the current row
    return np.array(sequences), np.array(labels)

# Create sequences and labels
X, y = create_sequences(df, seq_len=3)

# Normalize the non-categorical features
scaler = StandardScaler()
X[:, 3:] = scaler.fit_transform(X[:, 3:]) # Normalize features starting from index 3

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify to maintain class balance
)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, X.shape[1]) # Reshape for LSTM input
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # Add a dimension for binary classification
X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, X.shape[1])
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the LSTM-based model for reload prediction
class LSTMReloadNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # LSTM layer with 32 hidden units
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.ReLU(), # Activation function
            nn.Linear(32, 16), # Reduce 16 units
            nn.ReLU(), # Activation function
            nn.Linear(16, 1), # Output a single value
            nn.Sigmoid() # Sigmoid activation for binary classification
        )

    def forward(self, x):
        # Forward pass through the LSTM and fully connected layers
        _, (hn, _) = self.lstm(x) # Get the hidden state from the LSTM
        out = self.fc(hn.squeeze(0)) # Pass the hidden state through the fully connected layer
        return out

# Initialize the model, loss function, and optimizer
model = LSTMReloadNet(input_size=X.shape[1]) # Input size is the number of features in X
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.005) # Adam optimizer with a learning rate of 0.005

# Train the model
train_losses = [] # List to store training losses
for epoch in range(50): # Train for 50 epochs
    model.train() # Set the model to training mode
    optimizer.zero_grad() # Zero the gradients
    outputs = model(X_train) # Forward pass 
    loss = criterion(outputs, y_train) # Compute the loss
    loss.backward() # Backpropagation
    optimizer.step() # Update the weights
    train_losses.append(loss.item()) # Store the loss for this epoch

# Evaluate the model
model.eval() # Set the model to evaluation mode
with torch.no_grad(): # Disable gradient computation
    preds = model(X_test) # Get predictions for the test set 
    preds_class = (preds > 0.5).int() # Convert probabiliyies to binary predictions
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds_class, target_names=["no_reload", "reload"]))

    cm = confusion_matrix(y_test, preds_class) # Compute the confusion matrix
    fpr, tpr, _ = roc_curve(y_test, preds) # Compute the ROC curve
    auc = roc_auc_score(y_test, preds) # Compute the AUC score

# Save the trained model and scaler
torch.save(model.state_dict(), "reload_model.pt") # Save the model weights
joblib.dump(scaler, "scaler.save") # Save the scaler

# Plot training loss, confusion matrix and ROC curve.
plt.figure(figsize=(12, 4))

# Plot training loss
plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Plot confusion matrix
plt.subplot(1, 3, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["no", "yes"], yticklabels=["no", "yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# Plot ROC curve
plt.subplot(1, 3, 3)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--") # Diagonal line for random guessing
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

plt.tight_layout()
plt.show()
