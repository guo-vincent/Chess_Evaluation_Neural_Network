import timm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd

class ChessVisionTransformer(torch.nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(ChessVisionTransformer, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        
        self.model.patch_embed.img_size = (8, 8)
        
        # Adjust input channels to match the chess tensor
        self.model.patch_embed.proj = torch.nn.Conv2d(input_channels, 
                                                      self.model.patch_embed.proj.out_channels, 
                                                      kernel_size=(8, 8),
                                                      stride=(8, 8)) 
        
        # Custom positional embedding
        self.model.pos_embed = torch.nn.Parameter(torch.zeros(1, 2, self.model.embed_dim))
        
        # Adjust the classification head
        self.model.head = torch.nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class ChessPositionDataset(Dataset):
    def __init__(self, positions, labels):
        self.positions = positions
        self.labels = labels

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = self.positions[idx]
        label = self.labels[idx]
        return position, label.flatten()
    
def read_matrix(file_name, number_matrixes):
    # Rows to read: header + (number of matrices * 9 rows per matrix)
    rows_to_read = number_matrixes * 9 + 1

    data = pd.read_csv(file_name, nrows=rows_to_read)

    matrices = []
    for k in range(number_matrixes):
        start_row = k * 9

        # Extract the board matrix and evaluation from the DataFrame
        board_matrix = data.iloc[start_row:start_row+8, :-1].to_numpy(dtype=np.float32)
        evaluation = data.iloc[start_row+8, -1]

        matrices.append((board_matrix, evaluation))
    
    return matrices

if __name__ == "__main__":
    total_number = 12956364 # Number of matrices in the file
    white_matrices_total = 6473070
    black_matrices_total = 6483294
    file_name_white = R"CSVFiles/White.csv"
    file_name_black = R"CSVFiles/Black.csv"
    number_matrixes_white = 6473070    # Specify the number of matrices you want to read
    number_matrixes_black = 6483294    # Specify the number of matrices you want to read

    matrices_white = read_matrix(file_name_white, number_matrixes_white)

    X_white = np.array([matrix/100 for matrix, _ in matrices_white])
    y_white = np.array([evaluation for _, evaluation in matrices_white])

    # Convert evaluations
    y_white = y_white.reshape(-1, 1)
    print("Done")

    # Normalize evaluations using MinMaxScaler
    scaler = joblib.load(R"Scalers/ScalerWhite.pkl")
    normalized_y_white = scaler.transform(y_white).flatten()

    # Reshape X_white to include channel dimension
    X_white = X_white.reshape((X_white.shape[0], 1, 8, 8))  # (number_matrixes, height, width)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessVisionTransformer(input_channels=1, num_classes=1).to(device)

    criterion = torch.nn.L1Loss()  # aka Huber Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 95% train, 5% test
    X_train, X_test, y_train, y_test = train_test_split(X_white, normalized_y_white, test_size=0.05, random_state=42)

    # 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Datasets
    train_dataset = ChessPositionDataset(X_train, y_train)
    val_dataset = ChessPositionDataset(X_val, y_val)
    test_dataset = ChessPositionDataset(X_test, y_test)

    # Parameters:
    batch = 2048

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # Need this to transform the bias dtype to .float64 for bias dtype == input dtype.
    for positions, labels in train_loader:
        positions, labels = positions.to(device).float(), labels.to(device).float()

    for positions, labels in val_loader:
        positions, labels = positions.to(device).float(), labels.to(device).float()

    for positions, labels in test_loader:
        positions, labels = positions.to(device).float(), labels.to(device).float()

    # Early Stopping
    patience = 3
    best_val_loss = np.inf
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for positions, labels in train_loader:
            positions, labels = positions.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(positions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader):.4f}')

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for positions, labels in val_loader:
                positions, labels = positions.to(device), labels.to(device)
                outputs = model(positions)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Optionally save the best model
            torch.save(model.state_dict(), 'best_chess_vit_model_state_dict.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print('Early stopping')
                break

    # Optionally, load the best model for final evaluation
    model.load_state_dict(torch.load('best_chess_vit_model_state_dict.pth', weights_only=True))

    # Test the model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for positions, labels in test_loader:
            positions, labels = positions.to(device), labels.to(device)
            outputs = model(positions)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    print(f'Test Loss: {test_loss / len(test_loader):.4f}')
    

"""
# Loading the saved model
loaded_model = ChessVisionTransformer(input_channels=1, num_classes=1).to(device)
loaded_model.load_state_dict(torch.load('chess_vit_model_state_dict.pth'))
loaded_model.eval()
test_loss = 0.0
with torch.no_grad():
    for positions, labels in test_loader:
        positions, labels = positions.to(device), labels.to(device)
        outputs = model(positions)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}')
"""
