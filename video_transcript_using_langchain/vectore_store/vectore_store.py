from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

def initialize_vectorstore(file_path, collection_name, connection):
    """Initializes the vectorstore."""
    # Load documents
    loader = TextLoader(file_path=file_path)
    docs = loader.load() #The load method is called to read the documents from the file, returning a list of document objects.

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = PGVector(
        embeddings=embeddings, 
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    vectorstore.add_documents(documents=splits, embedding=embeddings)

    return vectorstore
