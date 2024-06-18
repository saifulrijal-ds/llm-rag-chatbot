import dotenv
from langchain_community.chat_models.pai_eas_endpoint import PaiEasChatEndpoint
from langchain_openai.chat_models import ChatOpenAI
from langchain_together import ChatTogether
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
import os

# Load environment variables from .env file
dotenv.load_dotenv()

REVIEWS_CHROMA_PATH = "chroma_data/"

# Initiate chat model
# chat_model = PaiEasChatEndpoint(
#     eas_service_url=os.environ["EAS_SERVICE_URL"],
#     eas_service_token=os.environ["EAS_SERVICE_TOKEN"],
#     temperature=0.1,
#     max_new_tokens=128
# )

# chat_model = ChatOpenAI(
#     model='gpt-3.5-turbo',
#     temperature=0.1,
#     max_tokens=256
# )

chat_model = ChatTogether(
    model='Qwen/Qwen2-72B-Instruct',
    temperature=0.1,
    max_tokens=256
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Create review messages template
# Include system and human template
review_template_str = """Your job is to use patient reviews to answer questions about their experience at a hospital. 
Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. 
If you don't know an answer, say you don't know.

{context}
"""
# System message prompt template
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['context'],
        template=review_template_str
    )
)

# Humana message prompt template
review_human_prompt =HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['context'],
        template="{question}"
    )
)

# Combine system and human message prompts
messages = [review_system_prompt, review_human_prompt]

# Create the chat prompt template
review_prompt_template = ChatPromptTemplate(
    input_variables=['context', 'question'],
    messages=messages
)

# Set up vector database to retrieved
reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=embeddings_model
)

# Retrieve with top 10 most similar or relevant
reviews_retriever = reviews_vector_db.as_retriever(k=10)

# Set the output parser
output_parser = StrOutputParser()

# Build chain with LCEL
# review_chain = review_prompt_template | chat_model | output_parser
review_chain = (
    {'context': reviews_retriever, 'question': RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | output_parser
)
