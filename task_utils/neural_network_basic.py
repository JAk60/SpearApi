import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class DeepNet(nn.Module):
    '''
        Class for Deep neural network, used in Joint learning class/Algorithm
        
    Args:
        input_size: number of features
        hidden_size: number of nodes in each of the two hidden layers
        output_size: number of classes
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepNet, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = nn.functional.relu(self.linear_1(x))
        out = nn.functional.relu(self.linear_2(out))
        return self.out(out)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracy = correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load the best model state (if early stopping was triggered)
    if best_model_state:
        model.load_state_dict(best_model_state)

    print('Training complete')
    return train_losses, val_losses

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    return test_loss

def plot_losses(train_losses, val_losses, num_epochs):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'go-', label='Validation loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.show()


def inference(model, data_loader):
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            probabilities.extend(probs.cpu().numpy())

    return predictions, probabilities

def test():
    # Example usage
    input_size = 10  # number of features
    hidden_size = 50  # number of nodes in each hidden layer
    output_size = 3  # number of classes

    model = DeepNet(input_size, hidden_size, output_size)

    # Sample data (replace with actual data)
    train_data = torch.randn(100, input_size)
    train_labels = torch.randint(0, output_size, (100,))
    val_data = torch.randn(20, input_size)
    val_labels = torch.randint(0, output_size, (20,))
    test_data = torch.randn(20, input_size)
    test_labels = torch.randint(0, output_size, (20,))

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    patience = 5  # Number of epochs to wait for improvement before stopping

    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience)
    test_loss = evaluate_model(model, test_loader, criterion)
    plot_losses(train_losses, val_losses, num_epochs)

    predictions, probabilities = inference(model, test_loader)
    print(f"Predictions: {predictions}\nProbabilities: {probabilities}")




if __name__ == "__main__":
    test()