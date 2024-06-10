import gradio as gr
from databricks.vector_search.client import VectorSearchClient
from databricks_genai_inference import ChatSession


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


CATALOG = "workspace"
DB='vs_demo_test'
VS_INDEX_NAME = 'fm_api_examples_vs_index_v3'
VS_INDEX_FULLNAME = f"{CATALOG}.{DB}.{VS_INDEX_NAME}"
VS_ENDPOINT_NAME = 'vs_endpoint'

vsc = VectorSearchClient(
    workspace_url="https://dbc-e3727d7d-82af.cloud.databricks.com",
    personal_access_token="dapi8596ab3c02ae6b5445f74e6ee8671afa"
)

# from databricks import sql

# Initialize model and index
chat_model = ChatSession(model="databricks-meta-llama-3-70b-instruct",
                         system_message="You are a helpful assistant. Answer the user's question based on the provided context.",
                         max_tokens=128)
index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME,
                      index_name=VS_INDEX_FULLNAME)
# index.sync()

def model_serving_endpoint(user_query):
    # Extract user query from request
    # user_query = request.get("query")
    
    # print(user_query)
    # Perform vector search to retrieve context
    raw_context = index.similarity_search(columns=["text", "title"],
                                          query_text=user_query,
                                          num_results=3)
    
    # Construct context string
    context_string = "Context:\n\n"
    for i, doc in enumerate(raw_context.get('result').get('data_array')):
        context_string += f"Retrieved context {i+1}:\n"
        context_string += doc[0]
        context_string += "\n\n"
    
    # Generate response using the model
    response = chat_model.reply(f"User question: {user_query}\n\nContext: {context_string}")
    
    return chat_model.last

# Example request handling
# request = {"query": "What are the available initiatives?"}
# response = model_serving_endpoint(request)
# print(response)


demo = gr.Interface(
    fn=model_serving_endpoint,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()