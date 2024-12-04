from py2neo import Graph, Node, Relationship
import csv
from tqdm import tqdm  # Import tqdm for progress bar

# Function to read the cleaned NER and sentiment results from CSV
def read_cleaned_data_from_csv(filename):
    cleaned_data = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cleaned_data.append(row)
    return cleaned_data

# Connect to the Neo4j database
def connect_to_neo4j(uri="bolt://localhost:7687", username="neo4j", password="12345678"):
    graph = Graph(uri, auth=(username, password))
    return graph

# Function to create a knowledge graph from cleaned data
def create_knowledge_graph(graph, cleaned_data):
    # Iterate over the cleaned data with a progress bar using tqdm
    for row in tqdm(cleaned_data, desc="Processing rows", unit="row"):
        post_title = row['Title']
        sentiment = row['Sentiment']

        # Attempt to convert 'Sentiment Score' to float, handle errors gracefully
        try:
            sentiment_score = float(row['Sentiment Score'])
        except ValueError:
            sentiment_score = None  # Set to None or any default value if conversion fails

        # Create the Post node with sentiment data
        post_node = Node("Post", title=post_title, sentiment=sentiment, sentiment_score=sentiment_score)
        graph.create(post_node)

        # Process entities from NER results
        entities = eval(row['Entities'])  # Convert string back to list of tuples
        for entity_name, entity_type in entities:
            # Create an Entity node
            entity_node = Node("Entity", name=entity_name, type=entity_type)
            graph.merge(entity_node, "Entity", "name")  # Avoid duplicates

            # Create a relationship between Post and Entity
            relationship = Relationship(post_node, "MENTIONS", entity_node)
            relationship["sentiment"] = sentiment
            relationship["sentiment_score"] = sentiment_score
            graph.create(relationship)

# Main execution
if __name__ == "__main__":
    # Step 1: Read the cleaned data from CSV
    input_filename = 'cleaned_ner_sentiment_results.csv'
    cleaned_data = read_cleaned_data_from_csv(input_filename)

    # Step 2: Connect to Neo4j
    graph = connect_to_neo4j(uri="bolt://localhost:7687", username="neo4j", password="12345678")

    # Step 3: Create the Knowledge Graph in Neo4j
    create_knowledge_graph(graph, cleaned_data)

    print("Knowledge graph created in Neo4j!")
