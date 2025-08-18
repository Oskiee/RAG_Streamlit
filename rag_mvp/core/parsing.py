from io import BytesIO
from typing import List, Any, Optional
import re
import os
from dataclasses import dataclass

import docx2txt
import chardet
from langchain.docstore.document import Document
import fitz
from hashlib import md5
from xlsx2html import xlsx2html
from pptx import Presentation
from openpyxl import load_workbook
import xlrd
import tempfile
import subprocess
import shutil

from abc import abstractmethod, ABC
from copy import deepcopy


@dataclass(init=False)
class File(ABC):
    """Represents an uploaded file comprised of Documents"""

    def __init__(
        self,
        name: str,
        id: str,
        metadata: Optional[dict[str, Any]] = None,
        docs: Optional[List[Document]] = None,
    ):
        self.name = name
        self.id = id
        self.metadata = metadata or {}
        self.docs = docs or []

    @classmethod
    @abstractmethod
    def from_bytes(cls, file: BytesIO) -> "File":
        """Creates a File from a BytesIO object"""

    def __repr__(self) -> str:
        return (
            f"File(name={self.name}, id={self.id},"
            f" metadata={self.metadata}, docs={self.docs})"
        )

    def __str__(self) -> str:
        return f"File(name={self.name}, id={self.id}, metadata={self.metadata})"

    def copy(self) -> "File":
        """Create a deep copy of this File"""
        return self.__class__(
            name=self.name,
            id=self.id,
            metadata=deepcopy(self.metadata),
            docs=deepcopy(self.docs),
        )


def strip_consecutive_newlines(text: str) -> str:
    """Strips consecutive newlines from a string
    possibly with whitespace in between
    """
    return re.sub(r"\s*\n\s*", "\n", text)

class DocxFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "DocxFile":
        filename = getattr(file, "name", "uploaded_file")
        ext = filename.split(".")[-1].lower()

        if ext == "doc":
            libreoffice_bin = (
                shutil.which("libreoffice")
                or shutil.which("soffice")
                or "/opt/homebrew/bin/soffice"
            )

            if not os.path.exists(libreoffice_bin):
                raise RuntimeError(
                    f"LibreOffice (soffice) не найден. Проверял: "
                    f"{shutil.which('libreoffice')}, {shutil.which('soffice')}, /opt/homebrew/bin/soffice"
                )

        file_bytes = file.read()
        file.seek(0)

        if ext == "docx":
            text = docx2txt.process(file)

        elif ext == "doc":
            with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                doc_path = tmp.name

            output_dir = tempfile.mkdtemp()
            subprocess.run(
                [libreoffice_bin, "--headless", "--convert-to", "docx", "--outdir", output_dir, doc_path],
                check=True
            )

            converted_path = os.path.join(output_dir, os.path.basename(doc_path) + "x")

            text = docx2txt.process(converted_path)

            os.remove(doc_path)
            if os.path.exists(converted_path):
                os.remove(converted_path)

        else:
            raise ValueError(f"Unsupported extension: {ext}")

        text = strip_consecutive_newlines(text)
        doc = Document(page_content=text.strip())
        doc.metadata["source"] = "p-1"

        return cls(
            name=filename,
            id=md5(file_bytes).hexdigest(),
            docs=[doc]
        )


class PdfFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "PdfFile":
        file.seek(0)
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        docs = []
        for i, page in enumerate(pdf):
            text = page.get_text(sort=True)
            # Detect encoding
            # Decode text using the detected encoding
            text = strip_consecutive_newlines(text)
            doc = Document(page_content=text.strip())
            doc.metadata["page"] = i + 1
            doc.metadata["source"] = f"p-{i + 1}"
            docs.append(doc)
        # file.read() mutates the file object, which can affect caching
        # so we need to reset the file pointer to the beginning
        file.seek(0)
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=docs)


class TxtFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "TxtFile":
        text = file.read().decode("utf-8", errors="replace")
        text = strip_consecutive_newlines(text)
        file.seek(0)
        doc = Document(page_content=text.strip())
        doc.metadata["source"] = "p-1"
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=[doc])

class XlsmFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "XlsmFile":
        docs = []
        ext = os.path.splitext(file.name)[1].lower()

        if ext in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
            workbook = load_workbook(file)
            for i, sheet in enumerate(workbook.sheetnames):
                file.seek(0)
                html = xlsx2html(file, sheet=sheet)
                html.seek(0)
                doc = Document(page_content=html.read().strip())
                doc.metadata["page"] = i + 1
                doc.metadata["source"] = f"p-{sheet}"
                docs.append(doc)

        elif ext in [".xls"]:
            workbook = xlrd.open_workbook(file_contents=file.read())
            for i, sheet_name in enumerate(workbook.sheet_names()):
                sheet = workbook.sheet_by_name(sheet_name)
                rows = []
                for r in range(sheet.nrows):
                    cells = "".join([f"<td>{sheet.cell_value(r, c)}</td>" for c in range(sheet.ncols)])
                    rows.append(f"<tr>{cells}</tr>")
                html_str = f"<table>{''.join(rows)}</table>"
                doc = Document(page_content=html_str.strip())
                doc.metadata["page"] = i + 1
                doc.metadata["source"] = f"p-{sheet_name}"
                docs.append(doc)

        else:
            raise ValueError(f"Unsupported Excel format: {ext}")

        file.seek(0)
        return cls(
            name=file.name,
            id=md5(file.read()).hexdigest(),
            docs=docs
        )


class PptxFile(File):
    @classmethod
    def from_bytes(cls, file: BytesIO) -> "PptxFile":
        presentation = Presentation(file)
        docs = []
        for slide_number, slide in enumerate(presentation.slides):
            text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"

            text = strip_consecutive_newlines(text)
            doc = Document(page_content=text.strip())
            doc.metadata["page"] = slide_number + 1
            doc.metadata["source"] = f"p-{slide_number + 1}"
            docs.append(doc)
            file.seek(0)
        return cls(name=file.name, id=md5(file.read()).hexdigest(), docs=docs)


def read_file(file: BytesIO) -> File:
    """Reads an uploaded file and returns a File object"""
    if file.name.lower().endswith(".docx") or file.name.lower().endswith(".doc"):
        return DocxFile.from_bytes(file)
    elif file.name.lower().endswith(".pdf"):
        return PdfFile.from_bytes(file)
    elif file.name.lower().endswith(".txt"):
        return TxtFile.from_bytes(file)
    elif file.name.lower().endswith(".xlsx") or file.name.lower().endswith(".xls") or file.name.lower().endswith(".xlsm"):
        return XlsmFile.from_bytes(file)
    elif file.name.lower().endswith(".pptx"):
        return PptxFile.from_bytes(file)
    else:
        raise NotImplementedError(f"File type {file.name.split('.')[-1]} not supported")
