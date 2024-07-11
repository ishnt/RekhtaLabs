from langchain_community.embeddings import OllamaEmbeddings

def print_embeddings(text):
    # Initialize Ollama embeddings with the llama3 model
    embeddings = OllamaEmbeddings(model='llama3:latest')
    
    # Get the embedding of the text
    query_result = embeddings.embed_query(text)
    
    # Print the embeddings
    print(f"Text: {text}")
    print(f"Embedding: {query_result}\n")

# Example usage:
text_input = "apple"

# Call the function to print the embeddings
print_embeddings(text_input)
