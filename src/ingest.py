import logging
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.getLogger("pypdf").setLevel(logging.ERROR)

CORPUS_DIR = Path("/Users/ajsalshereef/Library/CloudStorage/OneDrive-DeakinUniversity/Literature survey")
EXCLUDE_DIRS = {"Initial implementation"}


def load_documents():
    docs = []
    for pdf_path in sorted(CORPUS_DIR.rglob("*.pdf")):
        if any(part in EXCLUDE_DIRS for part in pdf_path.relative_to(CORPUS_DIR).parts):
            continue
        try:
            reader = PdfReader(str(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"  [skip] {pdf_path.name}: {e}")
            continue
        if not text.strip():
            print(f"  [skip] {pdf_path.name}: no extractable text (likely scanned images)")
            continue
        docs.append({
            "text": text,
            "file": pdf_path.name,
            "num_pages": len(reader.pages),
        })
    return docs


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece,
                "metadata": {"file": doc["file"], "chunk_index": i},
            })
    return chunks


if __name__ == "__main__":
    print("Loading PDFs (walking ~266 files, may take a few minutes)...")
    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents, {sum(d['num_pages'] for d in docs)} total pages")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size=500, overlap=50)")
    print("\nFirst chunk preview:")
    print(chunks[0]["text"][:200])
    print("metadata:", chunks[0]["metadata"])
