import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np

# Carregamento e limpeza dos dados
df = pd.read_csv("combined_log.csv")
df = df[
    (df["health"] <= 200) &
    (df["prev_health_3s"] <= 200) &
    (df["primary_ammo"] <= 200) &
    (df["secondary_ammo"] <= 200)
].copy()
df["reload_label"] = (df["action_took"] == "reload").astype(int)

# Codificar as ações anteriores (últimas 3)
action_encoder = LabelEncoder()
df["action_took_encoded"] = action_encoder.fit_transform(df["action_took"].fillna("hold"))

# Função para extrair janelas de sequência de ações e features
def create_sequences(dataframe, seq_len=3):
    sequences = []
    labels = []
    for i in range(seq_len, len(dataframe)):
        actions_seq = dataframe["action_took_encoded"].values[i - seq_len:i]
        features = dataframe.iloc[i][[
            "health", "primary_ammo", "primary_in_mag",
            "secondary_ammo", "secondary_in_mag",
            "prev_health_3s", "enemy_shown"
        ]].values.astype(float)
        full_seq = np.concatenate((actions_seq, features))
        sequences.append(full_seq)
        labels.append(dataframe.iloc[i]["reload_label"])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(df, seq_len=3)

# Normalização das features não categóricas
scaler = StandardScaler()
X[:, 3:] = scaler.fit_transform(X[:, 3:])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Converter para tensores
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, X.shape[1])
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, X.shape[1])
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# LSTM-based modelo
class LSTMReloadNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn.squeeze(0))
        return out

model = LSTMReloadNet(input_size=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Treinamento
train_losses = []
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

# Avaliação
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_class = (preds > 0.5).int()
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds_class, target_names=["no_reload", "reload"]))

    cm = confusion_matrix(y_test, preds_class)
    fpr, tpr, _ = roc_curve(y_test, preds)
    auc = roc_auc_score(y_test, preds)

# Gráficos
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["no", "yes"], yticklabels=["no", "yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.subplot(1, 3, 3)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

plt.tight_layout()
plt.show()
