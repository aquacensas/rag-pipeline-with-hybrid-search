'''Here we compare chunking strategies (fixed,structural,semnatic)
against the golden dataset and runn full eval suit.

Note on run-to-run variance: because generation uses a small non-zero
temperature, and correctness/ambiguity scoring is itself LLM-judged,
scores can shift slightly between identical runs.'''

from dataclasses import dataclass

from src.config import get_logger
from src.evaluation.golden_dataset import GoldenQA
from src.evaluation.metrics import run_eval, EvalReport

logger = get_logger(__name__)

STRATEGIES = ["fixed", "structural", "semantic"]

@dataclass
class StrategyComparisonReport:
    '''This holds one eval report per chunking startegy along with
    a simple summary table for side by side reading'''
    reports:dict[str,EvalReport]
    summary_table:list[dict]

def compare_chunking_strategies(dataset:list[GoldenQA])->StrategyComparisonReport:
    '''Runs the full golden dataset eval once per chunking strategy'''
    if not dataset:
        raise ValueError('Empty Dataset. Cannot perform strategy comparison.')

    reports: dict[str, EvalReport] = {}
    for strategy in STRATEGIES:
        logger.info(f'------- Running Evaluation against {strategy} chunking strategy --------')
        report = run_eval(dataset, strategy=strategy)
        reports[strategy]=report
        logger.info(
            f"'{strategy}' complete — correctness={report.mean_correctness:.3f}, "
            f"faithfulness={report.mean_faithfulness:.3f}, "
            f"retrieval_relevance={report.mean_retrieval_relevance:.3f}"
        )

    summary_table = [
        {
            "strategy": strategy,
            "mean_correctness": report.mean_correctness,
            "mean_faithfulness": report.mean_faithfulness,
            "mean_retrieval_relevance": report.mean_retrieval_relevance,
            "mean_ambiguity_handling": report.mean_ambiguity_handling,
            "refusal_accuracy": report.refusal_accuracy,
        }
        for strategy, report in reports.items()
    ]

    return StrategyComparisonReport(reports=reports, summary_table=summary_table)

def print_comparison_table(comparison:StrategyComparisonReport)->None:
    '''Prints a text table comparing all strategies'''

    headers = ["strategy", "correctness", "faithfulness", "retrieval_rel", "ambiguity", "refusal_acc"]
    print(f"{headers[0]:<12} {headers[1]:<12} {headers[2]:<13} {headers[3]:<14} {headers[4]:<10} {headers[5]:<12}")
    print("-" * 75)
    for row in comparison.summary_table:
        print(
            f"{row['strategy']:<12} "
            f"{row['mean_correctness']:<12.3f} "
            f"{row['mean_faithfulness']:<13.3f} "
            f"{row['mean_retrieval_relevance']:<14.3f} "
            f"{row['mean_ambiguity_handling']:<10.3f} "
            f"{row['refusal_accuracy']:<12.3f}"
        )
