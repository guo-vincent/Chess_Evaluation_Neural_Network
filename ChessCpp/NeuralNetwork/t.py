import timm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    rows_to_read = number_matrixes * 9 + 1

    data = pd.read_csv(file_name, nrows=rows_to_read)

    matrices = []
    for k in range(number_matrixes):
        start_row = k * 9

        board_matrix = data.iloc[start_row:start_row+8, :-1].to_numpy(dtype=np.float32)
        evaluation = data.iloc[start_row+8, -1]

        matrices.append((board_matrix, evaluation))
    
    return matrices

if __name__ == "__main__":
    file_name_white = R"CSVFiles/White.csv"
    number_matrixes_white = 6473070

    matrices_white = read_matrix(file_name_white, number_matrixes_white)

    X_white = np.array([matrix/100 for matrix, _ in matrices_white])
    y_white = np.array([evaluation for _, evaluation in matrices_white])

    y_white = y_white.reshape(-1, 1)

    scaler = joblib.load(R"Scalers/ScalerWhite.pkl")
    normalized_y_white = scaler.transform(y_white).flatten()

    X_white = X_white.reshape((X_white.shape[0], 1, 8, 8))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessVisionTransformer(input_channels=1, num_classes=1).to(device)

    # Loading the pre-trained model weights
    model.load_state_dict(torch.load('best_chess_vit_model_state_dict.pth'))
    model.eval()

    test_dataset = ChessPositionDataset(X_white, normalized_y_white)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    outputs_list = []
    labels_list = []

    with torch.no_grad():
        for positions, labels in test_loader:
            positions, labels = positions.to(device).float(), labels.to(device).float()
            outputs = model(positions).cpu().numpy()
            outputs_list.extend(outputs)
            labels_list.extend(labels.cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.scatter(labels_list, outputs_list, s=1, alpha=0.5)

    min_value = min(min(labels_list), min(outputs_list))
    max_value = max(max(labels_list), max(outputs_list))
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', linewidth=2, label="y = x")
    plt.xlabel("True Value")
    plt.ylabel("Estimated Model Predictions")
    plt.title("True Value vs Model Prediction")
    plt.legend()
    plt.show()
