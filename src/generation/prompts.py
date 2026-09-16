'''Prompt construction for grounded generation. This take chunks and a question, and produces
the exact text that gets sent to the LLM. No API call happens here pure string construction'''

from src.config import get_logger
logger=get_logger(__name__)

SYSTEM_PROMPT="""You are a documentation assistant. Answer the user's question using ONLY the numbered context blocks provided below. Follow these rules strictly:

1. Only use information explicitly stated in the context blocks. Do not use outside knowledge, even if you know the answer.
2. Cite every factual claim with the bracketed number of the context block that supports it, e.g. [1], [2]. Place citations immediately after the claim they support.
3. If multiple context blocks support the same claim, cite all of them, e.g. [1][3].
4. If the context does not contain enough information to answer the question fully, explicitly say so — do not guess or fill gaps with outside knowledge.
5. Keep your answer focused and avoid restating the context verbatim; synthesize it into a direct answer.

Context blocks:
{context_blocks}

Question: {question}

Answer (with inline citations):"""

def format_context_block(chunks:list[dict])->str:
    '''Turning a list of retrieved chunks into numbered context blocks the LLm can cite by number
    [1] [2] [3] ... This return a single string with each chunk labeled [1][2] etc including its source_path so 
    citations are traceable back to a real file'''

    if not chunks:
        raise ValueError('Cannot format context block from an empty chunk list')
    
    blocks=[]
    for i, chunk in enumerate(chunks,start=1):
        source=chunk.get('source_path','unknown source')
        content=chunk.get('content','').strip()
        blocks.append(f'[{i}] (Source: {source})\n{content}')

    return '\n\n'.join(blocks)

def build_generation_prompt(question:str,chunks:list[dict])->str:
    '''This assembles the full prompt sent to the LLM:system instructions + numbered context + user' question
    This returns a complete prompt string ready to sent to the LLM model'''

    if not question or not question.strip():
        raise ValueError('question cannot be empty')

    context_blocks=format_context_block(chunks)
    logger.info(f'Built generation prompt with {len(chunks)} context blocks')

    return SYSTEM_PROMPT.format(context_blocks=context_blocks,question=question)