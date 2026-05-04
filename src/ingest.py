import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "iot-knowledge")
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "iot_docs")


def load_documents():
    print(f"[1/5] Loading documents from: {os.path.abspath(DOCS_PATH)}")
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"      Loaded {len(docs)} document(s)")
    for doc in docs:
        print(f"       - {os.path.basename(doc.metadata.get('source', 'unknown'))}")
    return docs


def split_documents(docs):
    print("\n[2/5] Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"      Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"      Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    return chunks


def create_embeddings():
    print("\n[3/5] Initializing OpenAI embeddings (text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )
    print("      Embeddings model ready (dimension: 1536)")
    return embeddings


def setup_pinecone_index():
    print("\n[4/5] Setting up Pinecone index...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    print(f"      Existing indexes: {existing_indexes if existing_indexes else 'none'}")

    if INDEX_NAME not in existing_indexes:
        print(f"      Creating new index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        print(f"      Waiting for index to be ready...")
        while True:
            status = pc.describe_index(INDEX_NAME).status
            if getattr(status, "ready", None) or (isinstance(status, dict) and status.get("ready")):
                break
            print("      Still initializing...", end="\r")
            time.sleep(2)
        print(f"      Index '{INDEX_NAME}' is ready")
    else:
        print(f"      Index '{INDEX_NAME}' already exists, reusing it")

    return pc.Index(INDEX_NAME)


def upsert_to_pinecone(chunks, embeddings):
    print(f"\n[5/5] Upserting {len(chunks)} chunks to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    print(f"      Successfully upserted {len(chunks)} chunks")
    print(f"      Namespace: default")
    return vectorstore


def main():
    print("=" * 60)
    print("  IoT SLM RAG - Document Ingestion Pipeline")
    print("=" * 60)

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env file")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in .env file")

    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = create_embeddings()
    setup_pinecone_index()
    upsert_to_pinecone(chunks, embeddings)

    print("\n" + "=" * 60)
    print("  Ingestion complete! Your RAG knowledge base is ready.")
    print(f"  Index name: {INDEX_NAME}")
    print(f"  Total chunks indexed: {len(chunks)}")
    print("  Run: uvicorn src.api:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
