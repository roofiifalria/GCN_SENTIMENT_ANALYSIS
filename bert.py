import csv
import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Configure logger
logging.basicConfig(
    filename='bert_gcn_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Define a simple GCN model
class GCN(nn.Module):
    def _init_(self, input_dim, hidden_dim, output_dim):
        super(GCN, self)._init_()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        try:
            x = torch.relu(self.conv1(x))
            x = self.conv2(x)
            return x
        except Exception as e:
            logger.error(f"Error in forward pass of GCN: {e}")
            raise e

# Function to load dataset and create graph data using BERT embeddings
def create_graph_data(titles, tokenizer, model, device, batch_size=32):
    graphs = []
    csv_data = []

    for i in tqdm(range(0, len(titles), batch_size), desc="Processing Titles"):
        batch_titles = titles[i:i + batch_size]
        try:
            # Tokenize and move to device
            inputs = tokenizer(batch_titles, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_dim)

            for idx, title in enumerate(batch_titles):
                try:
                    embeddings = batch_embeddings[idx]  # (hidden_dim)

                    # Generate adjacency matrix (fully connected for simplicity)
                    num_nodes = 1  # Each title is treated as a single node
                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)  # No edges since 1 node

                    # Create graph data
                    graph = Data(
                        x=embeddings.unsqueeze(0),  # Add a batch dimension
                        edge_index=edge_index,
                        y=torch.tensor([1], device=device)  # Dummy label
                    )
                    graphs.append(graph)

                    # Prepare data for CSV (convert to CPU for saving)
                    csv_data.append({
                        "title": title,
                        "embedding": embeddings.cpu().numpy().tolist()
                    })
                except Exception as e:
                    logger.error(f"Error processing title '{title}': {e}")
        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")

    return graphs, csv_data

def train_and_evaluate_graphs(graphs, device, save_path_pth="model.pth", save_path_h5="model.h5"):
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    model = GCN(input_dim=768, hidden_dim=64, output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in train_loader:
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch.x, batch.edge_index)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                logger.error(f"Error during training batch: {e}")
        logger.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # Save the model in .pth format
    torch.save(model.state_dict(), save_path_pth)
    logger.info(f"Model saved in PyTorch format at: {save_path_pth}")

    # Save the model in .h5 format
    try:
        import h5py
        with h5py.File(save_path_h5, 'w') as h5_file:
            for name, param in model.named_parameters():
                h5_file.create_dataset(name, data=param.cpu().numpy())
        logger.info(f"Model saved in H5 format at: {save_path_h5}")
    except ImportError as e:
        logger.error("h5py is not installed. Unable to save model in H5 format.")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            try:
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())
            except Exception as e:
                logger.error(f"Error during evaluation batch: {e}")

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # Memilih perangkat (GPU atau CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer dan model BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    # Membaca data dari file CSV
    input_filename = 'cleaned_gcn.csv'
    titles = []
    try:
        with open(input_filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    titles.append(row[0])
    except Exception as e:
        logger.error(f"Error reading input file: {e}")

    # Membuat data graph dan embedding
    graphs, csv_data = create_graph_data(titles, tokenizer, model, device)

    # Menyimpan data graph ke file .pt
    torch.save(graphs, 'bert_graph_data.pt')
    logger.info("Graph data saved to 'bert_graph_data.pt'")

    # Menyimpan embedding ke file CSV
    output_csv_filename = 'bert_graph_embeddings.csv'
    try:
        with open(output_csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["title", "embedding"])
            writer.writeheader()
            writer.writerows(csv_data)
        logger.info(f"Graph embeddings saved to '{output_csv_filename}'")
    except Exception as e:
        logger.error(f"Error saving CSV data: {e}")

    # Melatih dan mengevaluasi model
    train_and_evaluate_graphs(
        graphs,
        device,
        save_path_pth="bert_gcn_model.pth",
        save_path_h5="bert_gcn_model.h5"
    )
    logger.info("BERT-based GCN sentiment analysis completed.")