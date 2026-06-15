import os
from pathlib import Path
import pymupdf
import pymupdf4llm

from app.config.settings import settings


BASE_DIR = Path(__file__).resolve().parents[2]

DOCS_DIR = Path(settings.pdf_dir)
MARKDOWN_DIR = Path(settings.markdown_dir)
PARENT_STORE_PATH = Path(settings.parent_store_path)

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)


def pdf_to_markdown(pdf_path: str | Path, output_dir: str | Path = MARKDOWN_DIR) -> Path:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)

    md = pymupdf4llm.to_markdown(
        doc,
        header=False,
        footer=False,
        page_separators=True,
        ignore_images=True,
        write_images=False,
        use_ocr=False,
    )

    md_cleaned = md.encode(
        "utf-8",
        errors="surrogatepass"
    ).decode(
        "utf-8",
        errors="ignore"
    )

    output_path = (output_dir / pdf_path.stem).with_suffix(".md")
    output_path.write_text(md_cleaned, encoding="utf-8")

    print(f"✓ Converted: {pdf_path.name} -> {output_path.name}")
    return output_path


def pdfs_to_markdowns(
    docs_dir: str | Path = DOCS_DIR,
    output_dir: str | Path = MARKDOWN_DIR,
    overwrite: bool = False,
) -> list[Path]:
    docs_dir = Path(docs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️ No PDF files found in: {docs_dir}")
        return []

    converted_files = []

    for pdf_path in pdf_files:
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")

        if md_path.exists() and not overwrite:
            print(f"⏭️ Skipped existing: {md_path.name}")
            converted_files.append(md_path)
            continue

        converted_files.append(
            pdf_to_markdown(pdf_path, output_dir)
        )

    return converted_files


if __name__ == "__main__":
    pdfs_to_markdowns(overwrite=False)