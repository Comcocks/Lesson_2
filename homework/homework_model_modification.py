import numpy
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from regression_basics.utils import make_regression_data, RegressionDataset, ClassificationDataset, accuracy


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.stop = False

    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False

        improved = value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            self.stop = self.counter >= self.patience

        return self.stop


class LinearRegression(nn.Module):
    def __init__(self, in_features, l1 = 0.0, l2 = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.L1 = l1
        self.L2 = l2
        self.early_stopping = EarlyStopping()

    def init_stopping(self, patience=10, min_delta=0.0):
        self.early_stopping = EarlyStopping(patience, min_delta)

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        l1 = 0.0
        l2 = 0.0

        for n, p in self.named_parameters():
            if 'weight' in n:
                l1 += p.abs().sum()
                l2 += (p**2).sum()

        return self.L1 * l1 + self.L2 * l2


class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes = 2):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits) if self.num_classes == 1 else torch.softmax(logits, dim=1)

    def predict(self, x):
        with torch.no_grad():
            return (self.predict_proba(x) > 0.5).int() if self.num_classes == 1 else torch.argmax(self.predict_proba(x), dim=1)


    def evaluate(self, X, y):
        self.eval()
        y_true = y.cpu().numpy().flatten()
        y_pred = self.predict(X).cpu().numpy().flatten()
        y_proba = self.predict_proba(X).cpu().numpy()

        return {
            'precision': self.precision(y_true, y_pred),
            'recall': self.recall(y_true, y_pred),
            'f1': self.f1(y_true, y_pred),
            'roc_auc': self.roc_auc(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    @staticmethod
    def precision(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        prec = []

        for i in range(n_classes):
            tp = cm[i, i]
            fp = numpy.sum(cm[:, i]) - tp
            prec.append(tp / (tp + fp) if tp + fp != 0 else 0.0)

        return numpy.mean(prec)

    @staticmethod
    def recall(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        rec = []

        for i in range(n_classes):
            tp = cm[i, i]
            fn = numpy.sum(cm[i, :]) - tp
            rec.append(tp / (tp + fn) if tp + fn != 0 else 0.0)

        return numpy.mean(rec)

    @staticmethod
    def f1(y_true, y_pred):
        prec = LogisticRegression.precision(y_true, y_pred)
        rec = LogisticRegression.recall(y_true, y_pred)

        return 2 * (prec * rec) / (prec + rec) if prec + rec != 0 else 0.0

    @staticmethod
    def roc_auc(y_true, y_proba):
        classes = numpy.unique(y_true)

        # Для бинарной
        if len(classes) == 2:
            if y_proba.shape[1] == 1:
                return LogisticRegression.binary_roc_auc(y_true, y_proba.ravel())
            else:
                return LogisticRegression.binary_roc_auc(y_true, y_proba[:, 1])

        # Для многоклассовой
        aucs = []
        for c in classes:
            y_score = y_proba[:, c]
            aucs.append(LogisticRegression.binary_roc_auc((y_true == c).astype(int), y_score))

        return numpy.mean(aucs)

    @staticmethod
    def binary_roc_auc(y_true, y_score):
        ind = numpy.argsort(y_score)[::-1]

        n_positive = numpy.sum(y_true)
        n_negative = len(y_true) - n_positive
        if n_positive == 0 or n_negative == 0:
            return 0.5

        tpr, fpr = [], []
        tp, fp = 0, 0

        prev_score = y_score[ind[0]] + 1
        for i in range(len(y_true[ind])):
            current_score = y_score[ind[i]]

            if current_score != prev_score:
                tpr.append(tp / n_positive)
                fpr.append(fp / n_negative)
                prev_score = current_score

            if y_true[ind][i] == 1:
                tp += 1
            else:
                fp += 1

        tpr.append(tp / n_positive)
        fpr.append(fp / n_negative)

        return sum(((fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1])) for i in range(1, len(fpr))) / 2


if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)

    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Создаём модель
    model = LinearRegression(in_features=1, l1=0.01, l2=0.01)
    model.init_stopping(patience=20, min_delta=0.001)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y) + model.regularization_loss()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        if model.early_stopping(avg_loss):
            print(f"Остановка на эпохе {epoch}")
            print("Лучший loss: ", model.early_stopping.best_value)
            break

        if epoch % 10 == 0:
            print(f"Эпоха {epoch}, loss: {avg_loss:.4f}")

    # Сохраняем модель
    torch.save(model.state_dict(), 'linreg_torch.pth')

    # Загружаем модель
    new_model = LinearRegression(in_features=1)
    new_model.load_state_dict(torch.load('linreg_torch.pth'))
    new_model.eval()

    # Логистическая регрессия
    # Генерируем данные
    X, y = make_classification(n_samples=400, n_features=20, n_classes=2, n_clusters_per_class=1, n_informative=15, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Создаём датасет и даталоадер
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=X.shape[1], num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_acc = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        if epoch % 10 == 0:
            metric = model.evaluate(X, y)

    metrics = model.evaluate(X, y.view(-1, 1))
    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # Визуализация confusion matrix
    disp = ConfusionMatrixDisplay(metrics['confusion_matrix'])
    disp.plot()
    plt.show()