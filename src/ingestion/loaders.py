'''This is to process directory of source document (Text, Markdown, PDFs, HTML)
reads everyhting adn strips it down to clean plain text and attaches metadata such as
source path and file type this is to make sure downstream chunking and embedding 
always works with the same shape of objects regarless of the orignal format'''

import re
from dataclasses import dataclass
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
import html as html_module

SKIP_DIRS = {"img", "js", "css", "images", "assets"}
SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".pdf"}

# Matches raw HTML tags like <span ...>, </font>, <u style="...">.
# Does NOT match markdown syntax (#, *, [], code fences) since those
# don't use angle brackets.
HTML_TAG_RE = re.compile(r"<[^>]+>")

@dataclass
class RawDocument:
    '''One loaded document normalised to plain text with meta data'''
    source_path:str  # path to original source file 
    file_type:str    # eg txt html pdf
    content:str      # the plain text content

def strip_embedded_html(text: str) -> str:
    text = HTML_TAG_RE.sub("", text)
    text = html_module.unescape(text)  
    return text

def read_markdown_or_text(path:Path)-> str:
    raw = path.read_text(encoding='utf-8', errors='ignore')
    return strip_embedded_html(raw)

def read_html(path:Path)->str:
    raw=path.read_text(encoding='utf-8',errors='ignore')
    soup=BeautifulSoup(raw,'html.parser')
    #get_text() strip all the tags, leaving behind just the readable text
    return soup.get_text(separator="\n")

def read_pdf(path:Path)->str:
    reader=PdfReader(str(path))
    pages_text=[page.extract_text() or "" for page in reader.pages]
    return '\n'.join(pages_text)

def load_document(path:Path, root:Path)-> RawDocument | None:
    '''Loading every single document into RawDocument, or None if unreadable/empty'''
    ext=path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None
    
    try:
        if ext in ('.md','.txt'):
            content=read_markdown_or_text(path)
        elif ext=='.html':
            content=read_html(path)
        elif ext=='.pdf':
            content=read_pdf(path)
        else:
            return None
    
    except Exception as e:
        print(f'[loader] Failed to read {path}:{e}')
        return None
    
    content=content.strip()
    if not content:
        return None #Skip all the empty files 
    
    return RawDocument(
        source_path=str(path.relative_to(root)),
        file_type=ext.lstrip('.'),
        content=content
    )

def load_corpus(raw_dir:str='data/raw')->list[RawDocument]:
    '''Run the raw directory recursively and load every supported file and skipping the static-asset folder'''
    root=Path(raw_dir)
    documents:list[RawDocument]=[]

    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
            
        doc=load_document(path,root)
        if doc is not None:
            documents.append(doc)
        
    return documents