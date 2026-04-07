"""Build the Chroma vector index from the hospital reviews CSV.

Run once (or whenever source data changes):

    uv run python scripts/build_index.py
"""

from pathlib import Path

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from hospital_rag.config import settings

REVIEWS_CSV = Path(__file__).parent.parent / "data" / "reviews.csv"


def build_index() -> None:
    loader = CSVLoader(file_path=str(REVIEWS_CSV), source_column="review")
    docs = loader.load()
    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(api_key=settings.openai_api_key),
        persist_directory=settings.chroma_path,
    )
    print(f"Built index: {len(docs)} documents → {settings.chroma_path!r}")


if __name__ == "__main__":
    build_index()
