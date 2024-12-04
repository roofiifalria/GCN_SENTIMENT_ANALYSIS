import csv
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Define a simple GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = torch.nn.Linear(input_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Function to load dataset and create graph data
# Placeholder for feature extraction and adjacency matrix creation
def create_graph_data(titles):
    graphs = []
    for title in titles:
        # Feature extraction (e.g., using embeddings or other vectorizations)
        features = torch.rand(len(title.split()), 100)  # Random features for illustration
        edge_index = torch.randint(0, len(features), (2, 50))  # Random edges for illustration

        # Create graph data
        graph = Data(x=features, edge_index=edge_index, y=torch.tensor([1]))  # Dummy label
        graphs.append(graph)
    return graphs

# Function to train and evaluate GCN model
def train_and_evaluate_graphs(graphs):
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    model = GCN(input_dim=100, hidden_dim=64, output_dim=3)  # Output_dim depends on sentiment classes
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):  # Number of epochs
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch.x, batch.edge_index)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch.y.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy}")

# Main execution
if __name__ == "__main__":
    # Step 1: Read titles from the CSV file
    input_filename = 'cleaned_gcn.csv'
    titles = []
    with open(input_filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # Check if the row is not empty
                titles.append(row[0])  # Assuming titles are in the first column

    # Step 2: Create graph data
    graphs = create_graph_data(titles)

    # Step 3: Train and evaluate the GCN model
    train_and_evaluate_graphs(graphs)

    print("GCN-based sentiment analysis completed.")
