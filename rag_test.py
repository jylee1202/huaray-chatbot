import argparse
import os
import sys
from pathlib import Path
from textwrap import shorten
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

from config import PINECONE_INDEX_NAME

TERM_DICTIONARY: List[Tuple[str, str]] = [
    ("초보자", "Beginner"),
    ("전문가", "Expert"),
    ("구루", "Guru"),
    ("프레임 속도", "Frame Rate"),
    ("프레임 수", "AcquisitionFrameCount"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query the GigE camera Pinecone index and inspect retrieved chunks.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask about the manual. If omitted, the script prompts via stdin.",
    )
    parser.add_argument(
        "--index",
        default=PINECONE_INDEX_NAME,
        help="Pinecone index name. Defaults to config setting.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Optional Pinecone namespace to query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve from Pinecone.",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=300,
        help="Maximum characters to show for each retrieved chunk preview.",
    )
    parser.add_argument(
        "--disable-answer",
        action="store_true",
        help="Skip calling the LLM for an answer; only show retrieved chunks.",
    )
    return parser


def load_environment() -> None:
    load_dotenv()
    missing = [env for env in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not os.environ.get(env)]
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def normalise_question(question: str) -> str:
    normalised = question
    for src, dest in TERM_DICTIONARY:
        normalised = normalised.replace(src, dest)
    return normalised


def get_vector_store(index_name: str, namespace: Optional[str]) -> PineconeVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_list = pinecone_client.list_indexes()
    if hasattr(index_list, "indexes"):
        existing_indexes = {idx.name for idx in index_list.indexes}
    else:
        existing_indexes = set(index_list)
    if index_name not in existing_indexes:
        raise ValueError(f"Pinecone index '{index_name}' does not exist.")
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )


def retrieve_chunks(
    vector_store: PineconeVectorStore,
    query: str,
    top_k: int,
):
    return vector_store.similarity_search_with_score(query, k=top_k)


def build_answer(question: str, documents: List[Document]) -> str:
    llm = ChatOpenAI(model="gpt-4o")
    context_sections = [
        f"[chunk {idx + 1}]\n{doc.page_content}" for idx, doc in enumerate(documents)
    ]
    context = "\n\n".join(context_sections)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 GigE/10GigE 에어리어 스캔 카메라 매뉴얼을 근거로 답변하는 전문가입니다. "
                "주어진 컨텍스트에서만 답변하고, 정보가 없으면 모른다고 답하세요. "
                "가능하면 관련 파라미터나 설정 이름을 언급하면서 2~3문장으로 설명하세요.",
            ),
            ("human", "질문: {question}\n\n컨텍스트:\n{context}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return response.content if hasattr(response, "content") else str(response)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    question = args.question or input("질문을 입력하세요: ").strip()
    if not question:
        print("질문이 비어 있습니다. 종료합니다.", file=sys.stderr)
        sys.exit(1)

    try:
        load_environment()
    except EnvironmentError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    normalised_question = normalise_question(question)

    try:
        vector_store = get_vector_store(args.index, args.namespace)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Pinecone 초기화 실패: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"원본 질문: {question}")
    if normalised_question != question:
        print(f"정규화된 질문: {normalised_question}")

    print("검색 중...")
    try:
        results = retrieve_chunks(vector_store, normalised_question, args.top_k)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"검색 실패: {exc}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("검색 결과가 없습니다.")
        sys.exit(0)

    retrieved_docs: List[Document] = []
    for idx, (doc, score) in enumerate(results, start=1):
        retrieved_docs.append(doc)
        source = doc.metadata.get("source", "")
        preview = shorten(doc.page_content, width=args.max_preview, placeholder="...")
        print(f"[score={score:.4f} | source={source}")
        print(preview)

    if args.disable_answer:
        return

"""    try:
        answer = build_answer(question, retrieved_docs)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"LLM 호출 실패: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=== 모델 응답 ===")
    print(answer)"""


if __name__ == "__main__":
    main()
