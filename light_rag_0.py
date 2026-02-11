import os
import asyncio
import aiohttp
from typing import List


class OllamaEmbeddingFunc:
    def __init__(self, base_url: str, model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        self.model = model

    async def __call__(self, texts: List[str]):
        url = f"{self.base_url}/api/embeddings"
        embeddings = []

        async with aiohttp.ClientSession() as session:
            for text in texts:
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                try:
                    async with session.post(url, json=payload) as response:
                        # Debug the response
                        content_type = response.content_type
                        print(f"Embedding response content-type: {content_type}")
                        print(f"Embedding response status: {response.status}")

                        if response.status == 200 and content_type == 'application/json':
                            result = await response.json()
                            embeddings.append(result.get('embedding', []))
                        else:
                            text_content = await response.text()
                            print(f"Embedding error response: {text_content}")
                            embeddings.append([])
                except Exception as e:
                    print(f"Embedding request error: {e}")
                    embeddings.append([])

        return embeddings


async def ollama_generate(base_url: str, model: str, prompt: str):
    url = base_url
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload) as response:
                print(f"Generation response status: {response.status}")
                print(f"Generation response content-type: {response.content_type}")

                if response.status == 200:
                    if response.content_type == 'application/json':
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        text_content = await response.text()
                        print(f"Unexpected response format: {text_content}")
                        return f"Error: Unexpected response format. Status: {response.status}"
                else:
                    error_text = await response.text()
                    return f"Error {response.status}: {error_text}"
        except Exception as e:
            return f"Connection error: {e}"


# Fixed version with proper self references
class SimpleLightRAG:
    def __init__(self, working_dir: str, ollama_url: str, model_name: str = "llama3"):
        self.working_dir = working_dir
        self.ollama_url = ollama_url.rstrip('/')  # Remove trailing slash
        self.model_name = model_name
        self.embedding_func = OllamaEmbeddingFunc(ollama_url)

        # Create working directory
        os.makedirs(working_dir, exist_ok=True)

        # Simple document store
        self.documents = []

    async def ainsert(self, text: str):
        """Insert text into the RAG system"""
        self.documents.append(text)
        print(f"Inserted document. Total documents: {len(self.documents)}")

    async def aquery(self, query: str, mode: str = "naive"):
        """Simple query implementation"""
        # For simplicity, just combine all documents as context
        context = "\n".join(self.documents)

        prompt = f"""Context information is below.
=====================
{context}
=====================
Given the context information and not prior knowledge, answer the query.
Query: {query}
Mode : {mode}
Answer:"""

        response = await self._ollama_generate(self.ollama_url, self.model_name, prompt)
        return response

    async def _ollama_generate(self, base_url: str, model: str, prompt: str):
        url = base_url
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        timeout = aiohttp.ClientTimeout(total=300)
        file_ollama_key = "ollama_api_key_file.txt"
        api_key_file = open(file_ollama_key)
        headers1 = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key_file.read()}"}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=headers1) as response:
                    print(f"Generation response status: {response.status}")

                    if response.status == 200:
                        if response.content_type == 'application/json':
                            result = await response.json()
                            return result.get('response', '')
                        else:
                            text_content = await response.text()
                            print(f"Response content: {text_content[:200]}...")  # First 200 chars
                            return f"Received non-JSON response. Check server logs."
                    else:
                        error_text = await response.text()
                        return f"HTTP Error {response.status}: {error_text}"
            except Exception as e:
                return f"Connection error: {e}"


# Test the connection separately
async def test_connection(base_url: str):
    """Test if we can connect to the Ollama server"""
    url = 'https://ollama.com/api/tags'

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                print(f"Connection test - Status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    print("Available models:", result.get('models', []))
                    return True
                else:
                    error_text = await response.text()
                    print(f"Connection failed: {error_text}")
                    return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False


# Usage example
async def main():
    OLLAMA_URL = "https://ollama.com/api/generate"  # CHANGE THIS!

    # First test connection
    # print("Testing connection...")
    # if not await test_connection(OLLAMA_URL):
    #     print("Cannot connect to Ollama server. Please check:")
    #     print("1. Server is running")
    #     print("2. URL is correct")
    #     print("3. Firewall/network settings")
    #     return

    # Initialize with remote Ollama
    rag = SimpleLightRAG(
        working_dir="./dickens",
        ollama_url=OLLAMA_URL,
        model_name="gpt-oss:120b-cloud"  # Make sure this model exists on your server
    )

    # Insert sample text
    print("Inserting document...")

    # Query
    print("Sending query...")

    # # feed the data
    # await rag.ainsert("Charles Dickens was an English writer and social critic. He created some of the world's best-known fictional characters and is regarded by many as the greatest novelist of the Victorian era.")
    # # ask query
    # query_result = await rag.aquery(query="Charles Dickens was from which era?", mode="hybrid")
    # # query_result = await rag.aquery(query="Was Charles Dickens using Google pay?", mode="local")
    # print("Result:", query_result)

    # file_topic1 = "data_in\\topic1_avatar.txt"
    # date_topic1 = open(file_topic1, 'r', encoding='utf-8').read()
    # # feed the data
    # await rag.ainsert(date_topic1)
    # # ask query
    # query_result = await rag.aquery(query="Tell me the plot of the movie in 1 line", mode="hybrid")
    # query_result = await rag.aquery(query="Tell me a spoiler alert about the gun used by them.", mode="hybrid")
    # print("Result:", query_result)
















    # file_topic1 = "data_in\\topic_2_needle_in_haystack.txt"
    # date_topic1 = open(file_topic1, 'r', encoding='utf-8').read()
    # # feed the data
    # await rag.ainsert(date_topic1)
    # # ask query
    # # query_result = await rag.aquery(query="Tell me a details about ethnic slurs, jokes.")
    # query_result = await rag.aquery(query="Tell me about ethnic slurs, jokes in short or quick to understand.")
    # print("Result:", query_result)

    # file_topic1 = "data_in\\topic3_pizza_recepie.txt"
    # date_topic1 = open(file_topic1, 'r', encoding='utf-8').read()
    # # feed the data
    # await rag.ainsert(date_topic1)
    # # ask query
    # query_result = await rag.aquery(query="List all spices mentioned in the text.")
    # print("Result:", query_result)

    # file_topic1 = "data_in\\topic4_Negative_Testing_AbsentInformation.txt"
    # date_topic1 = open(file_topic1, 'r', encoding='utf-8').read()
    # # feed the data
    # await rag.ainsert(date_topic1)
    # # ask query
    # query_result = await rag.aquery(query="What role did Shakespear play in the Battle of Cannae?.")
    # print("Result:", query_result)

    # file_topic1 = "data_in\\topic5_irrelevant_query_filtering.txt"
    # date_topic1 = open(file_topic1, 'r', encoding='utf-8').read()
    # # feed the data
    # await rag.ainsert(date_topic1)
    # # ask query
    # query_result = await rag.aquery(query="What is the capital of Happy Island?")
    # print("Result:", query_result)

    # file_topic1 = "data_in\\topic6_noise_stress_test.txt"
    # date_topic1 = open(file_topic1, 'r', encoding='utf-8').read()
    # # feed the data
    # await rag.ainsert(date_topic1)
    # # ask query
    # query_result = await rag.aquery(query="What is the launch date of news paper?")
    # print("Result:", query_result)

    # file_topic1 = "data_in\\topic7_distractor_entities.txt"
    # date_topic1 = open(file_topic1, 'r', encoding='utf-8').read()
    # # feed the data
    # await rag.ainsert(date_topic1)
    # # ask query
    # query_result = await rag.aquery(query="When was Neo from The Matrix born?")
    # print("Result:", query_result)


# Run the async function
if __name__ == "__main__":
    asyncio.run(main())