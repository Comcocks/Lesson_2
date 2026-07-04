import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import logging
import os


learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [16, 32, 64]
optimizers = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop
}
hyperparameters = {
    'learning_rate': learning_rates,
    'batch_size': batch_sizes,
    'optimizer': optimizers.keys(),
}


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def hyperparameters_experiment():
    # Случайный тензор для эксперимента
    X = torch.randn(1000, 10)
    y = X @ torch.randn(10, 1) + torch.randn(1000, 1) / 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Тестирование гиперпараметров")
    results = {}
    for opt_name, opt_class in optimizers.items():
        print()
        for lr in learning_rates:
            print()
            for bs in batch_sizes:
                print(f"Тестирование оптимизатора {opt_name} с lr:{lr} и bs:{bs}")
                key = f"{opt_name}_lr{lr}_bs{bs}"

                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)
                model = LinearRegression(10)
                criterion = nn.MSELoss()
                optimizer = opt_class(model.parameters(), lr=lr)

                losses = []
                for epoch in range(50):
                    epoch_loss = 0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        logits = model(batch_X)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    losses.append(epoch_loss / len(train_loader))
                results[key] = losses

    # Визуализация
    visualize_experiment_results(results, 'optimizer', '...', 0.01, 32)
    visualize_experiment_results(results, 'learning_rate', 'Adam', '...', 32)
    visualize_experiment_results(results, 'batch_size', 'Adam', 0.01, '...')


def visualize_experiment_results(results, hyperparam, opt_name, lr, bs):
    plt.figure(figsize=(10, 5))
    
    for n in hyperparameters[hyperparam]:
        if hyperparam != "learning_rate":
            key = f"{opt_name}_lr{lr}_bs{n}" if hyperparam == "batch_size" else f"{n}_lr{lr}_bs{bs}"
        else:
            key = f"{opt_name}_lr{n}_bs{bs}"
        plt.plot(results[key], label=f"{hyperparam}={n}")
        
    plt.title(f'Effect of {hyperparam} change (lr={lr}, batch={bs}, optimizer={opt_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/tests_{hyperparam}.png')
    plt.close()


def feature_engineering_experiment():
    X, y = make_regression(n_samples=1000, n_features=3, noise=15, random_state=42)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Базовая модель
    scaler_base = StandardScaler()
    train_model(scaler_base.fit_transform(X_train), y_train, scaler_base.transform(X_test), y_test, "BaseFeatures")

    # Модель с полиномиальными признаками
    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler_poly = StandardScaler()
    X_train_poly = scaler_poly.fit_transform(poly.fit_transform(X_train))
    X_test_poly = scaler_poly.transform(poly.transform(X_test))
    train_model(X_train_poly, y_train, X_test_poly, y_test, "PolynomialFeatures")

    # Взаимодействие признаков
    interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_inter = interactions.fit_transform(X_train)
    X_test_inter = interactions.transform(X_test)
    train_model(X_train_inter, y_train, X_test_inter, y_test, "FeaturesInteractions")

    # Статистические признаки
    stats = [[np.mean(n), np.var(n), np.sum(n)] for n in X_train]
    X_train_stats = np.hstack([X_train, np.array(stats)])
    stats = [[np.mean(n), np.var(n), np.sum(n)] for n in X_test]
    X_test_stats = np.hstack([X_test, np.array(stats)])
    train_model(X_train_stats, y_train, X_test_stats, y_test, "StatisticFeatures")


def train_model(X_train, y_train, X_test, y_test, name):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model = LinearRegression(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mse = criterion(pred, y_test).item()

    print(f"Model: {name}, MSE: {mse}")


if __name__ == "__main__":
    feature_engineering_experiment()
    hyperparameters_experiment()