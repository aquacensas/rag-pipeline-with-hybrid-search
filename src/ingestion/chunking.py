'''Chunking strategies fpr splittign the raw document into smaller chuck objects for 
embedding and retrieval. Three switchable strategies:
1) fixed -> cut every N characters with overlap
2) structural -> split along with the markdown heading (respects document sections)
3) semantic -> splitting where topic or meaning actually shifts'''

import re
from dataclasses import dataclass, field
from src.ingestion.loaders import RawDocument
import numpy as np


@dataclass
class Chunk:
    '''chunk tagged with where it came from and how it was produced'''
    chunk_id:str
    source_path:str
    chunk_index:str
    content:str
    strategy:str
    section_heading:str | None=None 
    char_count:int =field(init=False)

    def __post_init__(self):
        self.char_count=len(self.content)

# Strategy 1 fixed-size chunking

def chunk_fixed(doc:RawDocument, chunk_size: int=800, overlap: int=150)->list[Chunk]:
    '''cutting the document into fixed sized window with overlapping edges,
       so a sentence split across a boundary still appears whole in at least one chunk'''
    
    text=doc.content
    chunks:list[Chunk]=[]
    start=0
    index=0

    while start<len(text):
        end=start+chunk_size
        piece=text[start:end].strip()
        if piece:
            chunks.append(Chunk(
                chunk_id=f'{doc.source_path}::fixed::{index}',
                source_path=doc.source_path,
                chunk_index=index,
                content=piece,
                strategy='fixed'
            ))
            index+=1
        start+=chunk_size-overlap 
    
    return chunks

# Strategy 2 Structual Chunking around markdown headings 

HEADING_RE=re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)

def chunk_structural(doc:RawDocument, max_chunk_size:int=1200)->list[Chunk]:
    '''Splitting on markdown heading first, so each chunk respects a document's 
    actual section. If a section is long fall back to fixed sized splitting'''

    text=doc.content
    matches=list(HEADING_RE.finditer(text))

    # No Heading found, fall back to fixed sized chunking for the particular doc
    if not matches:
        return [
            Chunk(
                chunk_id=f'{c.chunk_id.replace('::fixed::',':structural::')}',
                source_path=c.source_path,
                chunk_index=c.chunk_index,
                content=c.content,
                strategy='structural'

            )
            for c in chunk_fixed(doc,chunk_size=max_chunk_size,overlap=0)
        ]

    chunks:list[Chunk]=[]
    index=0

    for i, match in enumerate(matches):
        heading=match.group(2).strip()
        section_start=match.start()
        section_end=matches[i+1].start() if i+1 < len(matches) else len(text)
        section_text=text[section_start:section_end].strip()

        if not section_text:
            continue
        
        if len(section_text)<=max_chunk_size:
            chunks.append(Chunk(
                chunk_id=f'{doc.source_path}::structual::{index}',
                source_path=doc.source_path,
                chunk_index=index,
                content=section_text,
                strategy='structural',
                section_heading=heading,
            ))
            index+=1
        else:
            # Section too long , split it further but kep the heading tagged
            start=0
            while start <len(section_text):
                piece=section_text[start:start + max_chunk_size].strip()
                if piece:
                    chunks.append(Chunk(
                        chunk_id=f'{doc.source_path}::structural::{index}',
                        source_path=doc.source_path,
                        chunk_index=index,
                        content=piece,
                        strategy='structural',
                        section_heading=heading,
                    ))
                    index+=1
                start+=max_chunk_size

    return chunks

# Structure 3: Semantic Chunking splitting where the topic shifts

def split_sentence(text:str)-> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]

def chunk_semantic(doc:RawDocument,embed_fn, similarity_threshold=0.5,max_chunk_size:int=1500,min_chunk_size:int=200)->list[Chunk]:
    '''Groups consecutive sentence together as logn as they stays semantically similar;
    starts new chunk when meaning shifts (drop cosine similarity) or text grows too long'''
    
    sentences = split_sentence(doc.content)
    if len(sentences) <= 1:
        return [
            Chunk(
                chunk_id=f"{doc.source_path}::semantic::0",
                source_path=doc.source_path,
                chunk_index=0,
                content=doc.content,
                strategy="semantic",
            )
        ] if doc.content.strip() else []

    embeddings = np.array(embed_fn(sentences))

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    chunks: list[Chunk] = []
    current_sentences = [sentences[0]]
    current_len = len(sentences[0])
    index = 0

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        candidate_len = current_len + len(sentences[i])

        topic_shifted = sim < similarity_threshold
        too_big = candidate_len > max_chunk_size
        big_enough_to_split = current_len >= min_chunk_size  # NEW guard

        if (topic_shifted and big_enough_to_split) or too_big:
            chunks.append(Chunk(
                chunk_id=f"{doc.source_path}::semantic::{index}",
                source_path=doc.source_path,
                chunk_index=index,
                content=" ".join(current_sentences),
                strategy="semantic",
            ))
            index += 1
            current_sentences = [sentences[i]]
            current_len = len(sentences[i])
        else:
            current_sentences.append(sentences[i])
            current_len = candidate_len

    if current_sentences:
        chunks.append(Chunk(
            chunk_id=f"{doc.source_path}::semantic::{index}",
            source_path=doc.source_path,
            chunk_index=index,
            content=" ".join(current_sentences),
            strategy="semantic",
        ))

    return chunks
def chunk_document(doc: RawDocument, strategy: str = "structural", **kwargs) -> list[Chunk]:
    """Single entry point — pick a strategy by name."""
    if strategy == "fixed":
        return chunk_fixed(doc, **kwargs)
    elif strategy == "structural":
        return chunk_structural(doc, **kwargs)
    elif strategy == "semantic":
        return chunk_semantic(doc, **kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")       


 

    

    

    




