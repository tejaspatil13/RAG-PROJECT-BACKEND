import json
import os
from datetime import datetime

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ( ContextualRecallMetric, ContextualPrecisionMetric,)

from modules.retriever import Retriever


load_dotenv(override=True)


# ==========================================
# CONFIGURATION
# ==========================================

GOLDEN_PATH = "goldens/retriever_goldens.json"

RESULTS_DIR = "evaluation_results"

JUDGE_MODEL = "gpt-4o-mini"

THRESHOLD = 0.7


# ==========================================
# CREATE RESULTS DIRECTORY
# ==========================================

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# LOAD GOLDEN DATASET
# ==========================================

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    goldens = json.load(f)


print(f"Loaded {len(goldens)} golden questions.")


# ==========================================
# CREATE RETRIEVER
# ==========================================

retriever = Retriever().get_retriever()


# ==========================================
# CREATE METRICS
# ==========================================

recall_metric = ContextualRecallMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=False,
)

precision_metric = ContextualPrecisionMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=False,
)


metrics = [
    recall_metric,
    precision_metric,
]


# ==========================================
# CREATE TEST CASES
# ==========================================

test_cases = []

for golden in goldens:

    query = golden["query"]

    documents = retriever.invoke(query)

    retrieval_context = [
        doc.page_content
        for doc in documents
    ]

    test_case = LLMTestCase(
        input=query,
        expected_output=golden["ideal_answer"],
        actual_output="Generator not evaluated",
        retrieval_context=retrieval_context,
    )

    test_cases.append(test_case)


# ==========================================
# RUN DEEPEVAL
# ==========================================

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
)


# ==========================================
# SAVE RESULTS
# ==========================================

print("\nEvaluation completed.")

print("Saving results...")


evaluation_results = []


for index, test_case in enumerate(test_cases):

    result = {
        "id": goldens[index]["id"],
        "query": test_case.input,

        "ideal_answer": test_case.expected_output,

        "expected_context": goldens[index].get(
            "expected_context",
            []
        ),

        "retrieval_context": test_case.retrieval_context,
    }

    evaluation_results.append(result)


# ==========================================
# SAVE BASIC RETRIEVAL RESULTS
# ==========================================

results_file = os.path.join(
    RESULTS_DIR,
    "retriever_results.json"
)


with open(results_file, "w", encoding="utf-8") as f:

    json.dump(
        evaluation_results,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(
    f"Saved retrieval results to: {results_file}"
)


# ==========================================
# SAVE EXPERIMENT CONFIGURATION
# ==========================================

summary = {

    "evaluation_date":
        datetime.now().isoformat(),

    "total_questions":
        len(goldens),

    "judge_model":
        JUDGE_MODEL,

    "threshold":
        THRESHOLD,

    "retriever":
        "chroma_similarity",

    "embedding_model":
        "gemini-embedding-001",

    "top_k":
        5,

    "golden_dataset":
        GOLDEN_PATH,

}


summary_file = os.path.join(
    RESULTS_DIR,
    "retriever_summary.json"
)


with open(summary_file, "w", encoding="utf-8") as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print(
    f"Saved evaluation summary to: {summary_file}"
)