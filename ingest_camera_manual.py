import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import PINECONE_INDEX_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk the GigE camera manual and upsert embeddings to Pinecone.",
    )
    parser.add_argument(
        "--pdf",
        default="GigE (10GigE) Area Scan Camera_User's Manual_V1.0.0.pdf",
        help="Path to the PDF manual to ingest.",
    )
    parser.add_argument(
        "--index",
        default=PINECONE_INDEX_NAME,
        help="Target Pinecone index name (defaults to config setting).",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Optional Pinecone namespace for the upsert operation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Character size for each text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between chunks.",
    )
    return parser


def ensure_env() -> str:
    load_dotenv()
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required for embeddings.")
    return pinecone_api_key


def load_documents(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def upsert_chunks(
    chunks,
    index_name: str,
    namespace: Optional[str],
    pinecone_api_key: str,
):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    client = Pinecone(api_key=pinecone_api_key)
    index_list = client.list_indexes()
    if hasattr(index_list, "indexes"):
        existing_indexes = {idx.name for idx in index_list.indexes}
    else:
        existing_indexes = set(index_list)
    if index_name not in existing_indexes:
        raise ValueError(f"Pinecone index '{index_name}' does not exist.")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    ids = vector_store.add_documents(chunks)
    return ids


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        pinecone_api_key = ensure_env()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(args.pdf)
    try:
        documents = load_documents(pdf_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not chunks:
        print("No chunks were produced from the PDF; aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(documents)} pages and produced {len(chunks)} chunks.")

    try:
        ids = upsert_chunks(
            chunks,
            index_name=args.index,
            namespace=args.namespace,
            pinecone_api_key=pinecone_api_key,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to upsert chunks: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Upserted {len(ids) if ids else 0} vectors to index '{args.index}'.")
    if ids:
        print(f"Sample ids: {ids[:5]}")


if __name__ == "__main__":
    main()
