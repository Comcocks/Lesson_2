import sklearn
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, filepath, target_column, task='regression', fit_scaler=True, scaler=None):
        self.df = pd.read_csv(filepath)
        self.target_column = target_column
        self.task = task
        self.scaler = scaler
        self.df = self.df.dropna()

        # Признаки и цели
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # Типы колонок
        self.numerics = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical = X.select_dtypes(exclude=[np.number, np.bool]).columns.tolist()

        # Кодирование категорий
        for col in self.categorical:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Кодирование для классификации
        if self.task == 'classification':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            y = y.astype(np.float32).reshape(-1, 1)
        else:
            y = y.values.astype(np.float32).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if fit_scaler:
            self.X = X_train.values.astype(np.float32)
            self.y = y_train
        else:
            self.X = X_test.values.astype(np.float32)
            self.y = y_test

        # Нормализация данных
        if scaler is not None and self.numerics:
            X[self.numerics] = self.scaler.fit_transform(X[self.numerics])

        self.X = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    from homework_model_modification import LinearRegression, LogisticRegression

    print("Boston Dataset:")
    train_reg = CustomDataset('data/BostonHousing.csv', 'tax', task='regression', fit_scaler=True)
    test_reg = CustomDataset('data/BostonHousing.csv', 'tax', task='regression', fit_scaler=False)
    train_loader = DataLoader(train_reg, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_reg, batch_size=32)

    model = LinearRegression(in_features=train_reg.X.shape[1], l1=0.01, l2=0.01)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"{epoch}: {loss.item():.4f}")
    torch.save(model.state_dict(), 'models/linreg_csv.pth')
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X)
            test_loss += criterion(logits, batch_y).item()
    print(f"Boston Housing MSE: {test_loss/len(test_loader):.4f}")

    print("\nTitanic Dataset:")
    train_reg = CustomDataset('data/Titanic-Dataset.csv', 'Survived', task='classification', fit_scaler=True)
    test_reg = CustomDataset('data/Titanic-Dataset.csv', 'Survived', task='classification', fit_scaler=False)
    train_loader = DataLoader(train_reg, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_reg, batch_size=32)

    num_classes = len(np.unique(train_reg.y))
    model = LogisticRegression(in_features=train_reg.X.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y.squeeze().long())
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'models/logreg_csv.pth')

    model.eval()
    correct, total = 0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X)
            proba = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            all_preds.append(predicted)
            all_true.append(batch_y.squeeze().long())
            total += batch_y.size(0)
            correct += (predicted == batch_y.squeeze().long()).sum().item()

    accuracy = correct / total
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_true).numpy()

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision:  {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:  {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1:  {f1_score(y_true, y_pred, average='weighted'):.4f}")