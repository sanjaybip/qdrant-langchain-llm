# Langchain and Qdrant for persistance vector store

This is very basic LLM app to demostrate how to use [qdrant](https://qdrant.tech/), a vector database (alternative to pinecone) to create persistant vector store.

We have used openAI as our LLM and a fictional story (in file story.txt) to create embedding and storing it into Qdrant permanently.

You need to add env variable in .env file (shown in .env.example).

Run the below command to install dependancy

```bash
pip install -r requirements.txt
```
