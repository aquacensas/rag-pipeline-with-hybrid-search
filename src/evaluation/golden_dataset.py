'''This loads and validate hand verified test cases fir evaluation process'''

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.config import get_logger

logger = get_logger(__name__)

class QuestionType(str,Enum):
    '''Types of question to evaluate on, each of that testing a different type of capabilities'''
    STRAIGHTFORWARD = "straightforward"    # single-source direct lookup
    MULTI_HOP = "multi_hop"               # requires combining 2+ documents
    NO_ANSWER = "no_answer"               # deliberately unanswerable from corpus
    AMBIGUOUS = "ambiguous"               # multiple reasonable interpretations

@dataclass
class GoldenQA:
    '''One hand verified testcase'''
    id:str
    question:str
    question_type:QuestionType
    golden_answer:str
    source_paths:list[str]
    notes:str="" 

def validate_entry(raw:dict,index:int)->GoldenQA:
    """Validate one entry and return a GoldenQA object or raise ValueError"""
    required_fields=["id","question","question_type","golden_answer"]
    missing =[f for f in required_fields if f not in raw]

    if missing:
        raise ValueError(f'entry at index {index} is missing fields {missing}')
    
    try:
        qtype = QuestionType(raw["question_type"])
    except ValueError:
        raise ValueError(
            f"Entry '{raw.get('id')}' has invalid question_type "
            f"'{raw['question_type']}' — must be one of {[t.value for t in QuestionType]}"
        )

    if qtype != QuestionType.NO_ANSWER and not raw["golden_answer"].strip():
        raise ValueError(
            f"Entry '{raw['id']}' has an empty golden_answer but is not "
            f"type 'no_answer' — every answerable question needs a real answer"
        )

    return GoldenQA(
        id=raw["id"],
        question=raw["question"],
        question_type=qtype,
        golden_answer=raw["golden_answer"],
        source_paths=raw["source_paths"],
        notes=raw.get("notes", ""),
    )

def load_golden_dataset(path:str='data/eval/golden_qa.json')->list[GoldenQA]:
    '''Loads and validates the golden qa dataset'''

    file_path=Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f'Golden dataset not found at path {path}. Please create it')

    with open(file_path,'r',encoding='utf-8') as f:
        raw_entries=json.load(f)
    
    dataset = [validate_entry(entry, i) for i, entry in enumerate(raw_entries)]

    ids=[entry.id for entry in dataset]
    if len(ids) != len(set(ids)):
        duplicates=[id_ for id_ in ids if ids.count(id_)>1]
        raise ValueError(f'Duplicate IDs found in golden dataset: {set(duplicates)}')

    logger.info(f"Loaded {len(dataset)} golden Q&A entries from {path}")

    type_counts = {t.value: sum(1 for e in dataset if e.question_type == t) for t in QuestionType}
    logger.info(f"Breakdown by type: {type_counts}")

    return dataset


