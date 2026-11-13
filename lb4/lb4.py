import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class paramRegression(nn.Module):
    def __init__(self, A, B, T, sigma):
        super(paramRegression, self).__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float32))
        self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        return self.A + self.B * torch.sin(2 * torch.pi * x / self.T + self.sigma)

    def fit(self, X, y, epochs=5000, batch_size=32):
        self.train()
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(X)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

    def predict(self, X, y):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X)
            return predictions, self.A.item(), self.B.item(), self.T.item(), self.sigma.item()

df = pd.read_csv('./lb4/data.csv', sep=';')
print(df.head(), df.columns)
X, y = df.drop(columns=['1HCOR_E']), df['1HCOR_E']

# преобразование в день месяца
X['start_date_time'] =pd.to_datetime(X['start_date_time']).dt.dayofyear

X, y = torch.tensor(X['start_date_time'], dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = paramRegression(A=1, B=4.0, T=40, sigma=0.0)
model.fit(X_train, y_train)
predictions, A, B, T, sigma = model.predict(X_test, y_test)
print(f'A: {A:.4f}, B: {B:.4f}, T: {T:.4f}, sigma: {sigma:.4f}')

L1 = nn.L1Loss()
loss = L1(predictions, y_test)
print(f'L1 Loss: {loss.item():.4f}')

# график
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test.numpy(), y=y_test.numpy(), label='Actual', color='blue')
sns.lineplot(x=X_test.numpy(), y=predictions.numpy(), label='Predicted', color='red')
plt.title('Actual vs Predicted')
plt.xlabel('Start Date Time (seconds since epoch)')
plt.ylabel('1HCOR_E')
plt.legend()
plt.show()

# все данные

pred, _, _, _, _ = model.predict(X, y)
loss_all = L1(pred, y)
print(f'L1 Loss (All Data): {loss_all.item():.4f}')
# baseline
avg = torch.mean(y)
baseline_loss = L1(avg.repeat(y.size(0)), y)
print(f'Baseline L1 Loss: {baseline_loss.item():.4f}')

avg_test = torch.mean(y_test)
baseline_loss_test = L1(avg_test.repeat(y_test.size(0)), y_test)
print(f'Baseline L1 Loss (Test): {baseline_loss_test.item():.4f}')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.numpy(), y=y.numpy(), label='Actual', color='blue')
sns.lineplot(x=X.numpy(), y=pred.numpy(), label='Predicted', color='red')
plt.title('Actual vs Predicted (All Data)')
plt.xlabel('Start Date Time (seconds since epoch)')
plt.ylabel('1HCOR_E')
plt.legend()
plt.show()
