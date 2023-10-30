from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import qdrant_client
import os

load_dotenv()

# create qdrant client
client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# create a new collection
vectors_config=qdrant_client.http.models.VectorParams(size=1536, distance=qdrant_client.http.models.Distance.COSINE)

client.create_collection(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    vectors_config=vectors_config
)

# create vectore store
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
vectore_store = Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
    embeddings=embeddings,
)

# add documents to vector store
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

with open('story.txt') as f:
    raw_texts = f.read()

texts = get_chunks(raw_texts)

vectore_store.add_texts(texts)

# add vector store into retreivel chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
    chain_type="stuff",
    retriever=vectore_store.as_retriever()
)

#test
query="What is the name of the main characters?"

response = qa.run(query)

print(response)