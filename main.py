import os
from dotenv import load_dotenv

# Standard imports from LangChain (though they might show deprecation warnings)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

def main():
    # 1. Load the PDF
    pdf_path = "/home/oxit8888/Projects/LangChain/langchain-learn/2210.03629v3.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split the document into smaller chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    # 3. Create vector embeddings for search
    embeddings = OpenAIEmbeddings()

    # 4. Initialize (or create) a local FAISS vector index
    # If an index does not exist yet, build it
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    # Load an existing index (useful for bigger projects)
    # new_vectorstore = FAISS.load_local(
    #     "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    # )
    # If you already saved the index, uncomment the code above and comment out the line below
    new_vectorstore = vectorstore

    # 5. Initialize memory for storing conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # the key under which messages are stored
        return_messages=True        # whether to return the message list
    )

    # 6. Create a ConversationalRetrievalChain
    #    which can take conversation history into account + search the FAISS index
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0),
        retriever=new_vectorstore.as_retriever(),
        memory=memory,
        # you can fine-tune parameters here (e.g., max_tokens, etc.)
    )

    print("Chat with the PDF document. Type 'exit' to leave the chat.")
    while True:
        query = input("\nYou: ")
        if query.lower().strip() in ["exit", "quit", "bye"]:
            print("Exiting chat...")
            break

        # 7. Call the chain with the user's query
        result = qa_chain({"question": query})

        # 8. Print the response
        #    'result' includes keys like {"answer": "...", "chat_history": [...], etc.}
        print(f"Bot: {result['answer']}")

if __name__ == "__main__":
    main()
