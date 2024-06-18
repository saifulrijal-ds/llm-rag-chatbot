import dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_together.embeddings import TogetherEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

reviews_vector_db = Chroma.from_documents(
    reviews, embeddings_model, persist_directory=REVIEWS_CHROMA_PATH
)
