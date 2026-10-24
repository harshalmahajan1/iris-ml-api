import torch
import joblib
from torch import nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

torch.manual_seed(42)

# 1. Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data before scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features without leaking test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "iris_scaler.joblib")
print("Scaler saved successfully!")

# 4. Convert NumPy arrays into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 5. Build a neural network
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. Train the model
for epoch in range(200):
    model.train()

    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 7. Evaluate on unseen test data
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    predictions = test_logits.argmax(dim=1)

accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
print(f"Test accuracy: {accuracy:.4f}")

# 8. Save the trained model
torch.save(model.state_dict(), "iris_pytorch_model.pth")
print("PyTorch model saved successfully!")
