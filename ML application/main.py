import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt
import os

def one_hot_encode(arr, n_labels):
    arr = arr.cpu().numpy()  # GPU 텐서를 CPU로 이동하고 numpy 배열로 변환
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return torch.tensor(one_hot, dtype=torch.float32)

def train(model, trn_loader, device, criterion, optimizer, dataset):
    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = torch.tensor(one_hot_encode(inputs, len(dataset.chars)), dtype=torch.float).to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):
            hidden = tuple([h.to(device) for h in hidden])  # LSTM의 hidden 상태를 GPU로 이동
        else:
            hidden = hidden.to(device)
        output, _ = model(inputs, hidden)
        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion, dataset):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = torch.tensor(one_hot_encode(inputs, len(dataset.chars)), dtype=torch.float).to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple([h.to(device) for h in hidden])  # LSTM의 hidden 상태를 GPU로 이동
            else:
                hidden = hidden.to(device)
            output, _ = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def save_loss_plot(train_losses, val_losses, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label=f'{model_name} Train Loss')
    plt.plot(val_losses, label=f'{model_name} Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Loss Over Epochs')
    plt.savefig(f'./hw3/{model_name}_loss_plot.png')
    plt.close()

def main():
    # # Ensure the model directory exists
    # if not os.path.exists('./model'):
    #     os.makedirs('./model')

    batch_size = 64
    seq_length = 30
    epochs = 50
    lr = 0.001
    patience = 5  # early stopping을 위한 patience 설정

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Shakespeare('./hw3/data/shakespeare_train.txt')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    vocab_size = len(dataset.chars)
    hidden_size = 256  # hidden node 크기 감소
    n_layers = 2

    # RNN 학습 및 저장
    rnn_model = CharRNN(vocab_size, hidden_size, vocab_size, n_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=lr)

    rnn_train_losses = []
    rnn_val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        rnn_train_loss = train(rnn_model, train_loader, device, criterion, rnn_optimizer, dataset)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion, dataset)
        rnn_train_losses.append(rnn_train_loss)
        rnn_val_losses.append(rnn_val_loss)

        if rnn_val_loss < best_val_loss:
            best_val_loss = rnn_val_loss
            epochs_no_improve = 0
            torch.save(rnn_model.state_dict(), './hw3/model/rnn_model_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        print(f"Epoch {epoch+1}/{epochs}, RNN Train Loss: {rnn_train_loss:.4f}, RNN Val Loss: {rnn_val_loss:.4f}")

    torch.save(rnn_model.state_dict(), './hw3/model/rnn_model_best.pth')
    save_loss_plot(rnn_train_losses, rnn_val_losses, 'RNN')

    # LSTM 학습 및 저장
    lstm_model = CharLSTM(vocab_size, hidden_size, vocab_size, n_layers).to(device)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=lr)

    lstm_train_losses = []
    lstm_val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        lstm_train_loss = train(lstm_model, train_loader, device, criterion, lstm_optimizer, dataset)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion, dataset)
        lstm_train_losses.append(lstm_train_loss)
        lstm_val_losses.append(lstm_val_loss)

        if lstm_val_loss < best_val_loss:
            best_val_loss = lstm_val_loss
            epochs_no_improve = 0
            torch.save(lstm_model.state_dict(), './hw3/model/lstm_model_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        print(f"Epoch {epoch+1}/{epochs}, LSTM Train Loss: {lstm_train_loss:.4f}, LSTM Val Loss: {lstm_val_loss:.4f}")

    torch.save(lstm_model.state_dict(), './hw3/model/lstm_model_best.pth')
    save_loss_plot(lstm_train_losses, lstm_val_losses, 'LSTM')

if __name__ == '__main__':
    main()
