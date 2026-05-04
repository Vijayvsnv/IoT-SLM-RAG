import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "iot-knowledge")

SYSTEM_PROMPT = (
    "You are an IoT domain expert AI assistant with deep knowledge of IoT systems, "
    "protocols, devices, and troubleshooting. Your knowledge covers IoT protocols "
    "(MQTT, CoAP, HTTP, Zigbee, LoRaWAN, BLE), microcontrollers (ESP32, ESP8266, "
    "Arduino, Raspberry Pi), sensors (DHT11, DHT22, LM35, PIR, ultrasonic), and "
    "common errors. Always give practical device-specific answers with code examples "
    "in Arduino C++ or MicroPython. When context is provided from knowledge base, "
    "cite it as 'According to your knowledge base...'. If context is insufficient, "
    "answer from general IoT expertise and mention 'Based on general IoT knowledge...'."
    "\n\n"
    "Use the following retrieved context to answer the question:\n"
    "{context}"
)


def build_rag_chain():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def ask(chain, question: str) -> dict:
    result = chain.invoke({"input": question})

    sources = []
    for doc in result.get("context", []):
        source = {
            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
        }
        if source not in sources:
            sources.append(source)

    return {
        "answer": result["answer"],
        "sources": sources,
    }


if __name__ == "__main__":
    print("Building RAG chain...")
    chain = build_rag_chain()
    print("RAG chain ready. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = ask(chain, question)
        print(f"\nAssistant: {result['answer']}")
        print(f"\nSources:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['source']}: {src['content'][:100]}...")
        print()
