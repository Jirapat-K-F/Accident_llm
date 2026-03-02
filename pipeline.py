import csv
import os

import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

import config
from utils import Reader, Indexing
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

# ---------------------------------------------------------------------------
# DeepEval timeout configuration
# ---------------------------------------------------------------------------
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
os.environ["DEEPEVAL_TOTAL_TIMEOUT_SECONDS_OVERRIDE"] = "600"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISABLED_EVALUATION: dict = {
    metric: {"score": "Disabled", "reason": "Evaluation was disabled"}
    for metric in ("AnswerRelevancy", "ContextualPrecision", "ContextualRecall", "ContextualRelevancy")
}

FAILED_CONNECTION_EVALUATION: dict = {
    metric: {"score": "N/A", "error": "Ollama connection failed"}
    for metric in ("AnswerRelevancy", "ContextualPrecision", "ContextualRecall", "ContextualRelevancy")
}

PROMPT_TEMPLATE = """คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์อุบัติเหตุและการเสนอมาตรการแก้ไขปัญหา

กรุณาเสนอมาตรการแก้ไขปัญหาระยะสั้นที่เหมาะสมต่อเหตุการณ์ในรายงานสืบสวนอุบัติเหตุเชิงลึกต่อไปนี้ โดยอ้างอิงจากกรณีที่คล้ายคลึงกันและมาตรการที่ใช้ในกรณีเหล่านั้น:

=== เหตุการณ์ที่ต้องวิเคราะห์ ===
{validation_case}

=== กรณีที่คล้ายคลึงและมาตรการที่ใช้ (สำหรับอ้างอิง) ===
{similar_cases}

=== คำแนะนำ ===
ปรับมาตรการให้เหมาะกับลักษณะเฉพาะของกรณีนี้
บอกแค่มาตรการระยะสั้นเท่านั้น
ระบุผู้รับผิดชอบสำหรับแต่ละมาตรการ
หากไม่สามารถหาแนวทางแก้ไขได้จากกรณีที่คล้ายคลึง สามารถตอบสั้นๆ ได้ว่า "ไม่พบมาตรการที่เหมาะสม"
กรุณาให้ผลลัพธ์เป็นมาตรการที่ชัดเจน เป็นระบบ และสามารถนำไปปฏิบัติได้จริง แสดงผลลัพธ์แค่มาตรการที่แนะนำเท่านั้นโดยไม่ต้องใส่ข้อมูลอื่นเพิ่มเติม:
"""


class Pipeline:
    def __init__(
        self,
        chroma_db_path: str,
        full_docs_directory: str,
        k: int = 5,
        model_name: str = "llama3",
        collection_name: str = "accident_report_summaries",
        enable_evaluation: bool = True,
    ):
        self.chroma_db_path = chroma_db_path
        self.full_docs_directory = full_docs_directory
        self.k = k
        self.enable_evaluation = enable_evaluation
        self.template = PROMPT_TEMPLATE

        self.reader = Reader()
        self.indexing = Indexing(model_name=model_name)
        self.llm = OllamaLLM(model=model_name)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_collection(collection_name)

        # Initialize local judge for evaluation
        # FIX: use explicit Exception instead of bare except
        self.local_judge = None
        if self.enable_evaluation:
            self.local_judge = self._init_local_judge()

        # Pre-build evaluation metrics once (avoid re-creating on every call)
        # FIX: metrics were previously re-initialised inside evaluate() on every call
        if self.local_judge is not None:
            self._build_metrics()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_local_judge(self) -> OllamaModel | None:
        """Try to instantiate OllamaModel; fall back gracefully."""
        for model_tag in ("llama3.1:latest", "deepseek-r1:8b"):
            try:
                judge = OllamaModel(model=model_tag, base_url="http://localhost:11434")
                print(f"✓ Local judge initialised with model: {model_tag}")
                return judge
            except Exception as exc:  # FIX: was bare `except:`
                print(f"  ✗ Could not initialise judge with {model_tag}: {exc}")
        print("✗ All judge models failed — evaluation will be skipped")
        return None

    def _build_metrics(self) -> None:
        """Instantiate DeepEval metrics once and store as instance attributes."""
        self.relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=self.local_judge)
        self.precision_metric = ContextualPrecisionMetric(threshold=0.5, model=self.local_judge)  # FIX: typo presision → precision
        self.recall_metric = ContextualRecallMetric(threshold=0.5, model=self.local_judge)
        self.contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.5, model=self.local_judge)

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """
        Convert ChromaDB distance to a 0-1 similarity score.

        FIX: the original `1 - dist` only works for cosine distance (range 0-1).
        ChromaDB defaults to L2 (Euclidean) distance which is unbounded,
        so `1 - dist` can produce negative values.
        This formula maps any non-negative distance to (0, 1].
        """
        return 1.0 / (1.0 + distance)

    def _disabled_evaluation(self) -> dict:
        return DISABLED_EVALUATION.copy()

    # ------------------------------------------------------------------
    # Core pipeline steps
    # ------------------------------------------------------------------

    def find_top_k(self, val_summary: str) -> list[dict]:
        print(f"Finding {self.k} similar files from ChromaDB...")

        results = self.collection.query(query_texts=[val_summary], n_results=self.k)

        distances = results["distances"][0]
        ids_result = results["ids"][0]

        similar_files = []
        for rank, (doc_id, dist) in enumerate(zip(ids_result, distances), start=1):
            sim_score = self._distance_to_similarity(dist)  # FIX: was `1 - dist`
            similar_files.append(
                {"rank": rank, "filename": doc_id, "similarity_score": round(sim_score, 4)}
            )
            print(f"  {rank}. {doc_id} (similarity: {round(sim_score, 4)})")

        print(f"✓ Found {len(similar_files)} similar files")
        return similar_files

    def get_full_documents(self, similar_files: list[dict]) -> dict[str, str]:
        print("Retrieving full documents from directory...")

        full_documents = {}
        for file_info in similar_files:
            filename = file_info["filename"]
            file_path = os.path.join(self.full_docs_directory, filename)

            if not os.path.exists(file_path):
                print(f"  ✗ File not found: {file_path}")
                continue

            try:
                content = self.reader.read(file_path)
                if content.startswith("Error reading file"):
                    print(f"  ✗ Failed to read: {filename}")
                else:
                    full_documents[filename] = content
                    print(f"  ✓ Retrieved: {filename}")
            except Exception as exc:
                print(f"  ✗ Error reading {filename}: {exc}")

        print(f"✓ Retrieved {len(full_documents)} full documents")
        return full_documents

    def get_prediction(self, val_content: str, similar_cases_context: str) -> tuple[str, str]:
        """
        Run the LLM chain and return (prediction, rendered_prompt).

        The rendered prompt is returned so it can be passed as `input` to
        LLMTestCase — giving the DeepEval judge the exact query the LLM saw,
        with all template variables filled in (not raw {placeholders}).
        """
        rendered_prompt = self.template.format(
            validation_case=val_content,
            similar_cases=similar_cases_context,
        )
        prompt = ChatPromptTemplate.from_template(self.template)
        chain = prompt | self.llm
        prediction = chain.invoke(
            {"validation_case": val_content, "similar_cases": similar_cases_context}
        )
        return prediction, rendered_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, val_file_path: str, gt_file_path: str) -> dict:
        """Process a single validation file and return results."""
        val_content = self.reader.read(val_file_path)
        val_summary = self.indexing.summarize_with_llm(val_content)
        print("Validation Content Loaded.")

        similar_files = self.find_top_k(val_summary)
        full_documents = self.get_full_documents(similar_files)
        similar_cases_context = "\n\n".join(full_documents.values())

        prediction, rendered_prompt = self.get_prediction(val_content, similar_cases_context)

        gt = self.reader.read(gt_file_path)

        if self.enable_evaluation and self.local_judge is not None:
            evaluation_results = self.evaluate(
                rendered_prompt=rendered_prompt,
                predicted_measures=prediction,
                ground_truth=gt,
                retrieved_docs=list(full_documents.values()),
            )
        else:
            print("Evaluation disabled or local judge not available. Skipping...")
            evaluation_results = self._disabled_evaluation()

        result = {
            "val": os.path.basename(val_file_path),
            "TopK": "\n".join(
                f"{f['rank']}. {f['filename']} (similarity: {f['similarity_score']})"
                for f in similar_files
            ),
            "results": prediction,
            "evaluation": evaluation_results,
        }

        self.write_to_csv([result], os.path.join("res", "eval", "single_file_results.csv"))
        return result

    def predict_folder(self, val_folder_path: str, gt_folder_path: str) -> list[dict]:
        """Process all .docx validation files in a folder."""
        print(f"\n=== Processing folder: {val_folder_path} ===")

        val_files = sorted(f for f in os.listdir(val_folder_path) if f.endswith(".docx"))[:5]

        if not val_files:
            print("No .docx files found in validation folder")
            return []

        all_results = []

        # FIX: removed hard-coded [:5] slice — now processes all files
        for i, val_file in enumerate(val_files, start=1):
            print(f"\n--- Processing file {i}/{len(val_files)}: {val_file} ---")

            val_file_path = os.path.join(val_folder_path, val_file)
            gt_file_path = os.path.join(gt_folder_path, val_file)

            if not os.path.exists(gt_file_path):
                print(f"  Warning: GT file not found for {val_file}, skipping...")
                continue

            try:
                val_content = self.reader.read(val_file_path)
                val_summary = self.indexing.summarize_with_llm(val_content)

                similar_files = self.find_top_k(val_summary)
                full_documents = self.get_full_documents(similar_files)
                similar_cases_context = "\n\n".join(full_documents.values())

                prediction, rendered_prompt = self.get_prediction(val_content, similar_cases_context)
                gt = self.reader.read(gt_file_path)

                if self.enable_evaluation and self.local_judge is not None:
                    evaluation_results = self.evaluate(
                        rendered_prompt=rendered_prompt,
                        predicted_measures=prediction,
                        ground_truth=gt,
                        retrieved_docs=list(full_documents.values()),
                        original_content=val_content,
                        similar_cases_context=similar_cases_context,
                    )
                else:
                    evaluation_results = self._disabled_evaluation()

                result = {
                    "val": val_file,
                    "TopK": "\n".join(
                        f"{f['rank']}. {f['filename']} (similarity: {f['similarity_score']})"
                        for f in similar_files
                    ),
                    "results": prediction,
                    "evaluation": evaluation_results,
                }
                all_results.append(result)
                print(f"✓ Successfully processed {val_file}")

            except Exception as exc:
                print(f"✗ Error processing {val_file}: {exc}")
                continue

        if all_results:
            output_csv = os.path.join("res", "eval", "folder_validation_results.csv")
            self.write_to_csv(all_results, output_csv)
            print(f"\n✓ Processed {len(all_results)}/{len(val_files)} files successfully")
            print(f"✓ Consolidated results saved to {output_csv}")

        return all_results

    def evaluate(
        self,
        rendered_prompt: str,      # Fully rendered prompt (template + filled vars) — the real "question"
        predicted_measures: str,   # LLM-generated answer
        ground_truth: str,         # Expected / gold-standard answer
        retrieved_docs: list[str], # Top-K full documents used as context
        original_content: str = "",       # Original validation content (optional, can be used in evaluation reasoning
        similar_cases_context: str = "", # The combined context of similar cases (optional, can be used in evaluation reasoning)
    ) -> dict:
        """
        Run all four DeepEval RAG metrics and return a score/reason dict.

        FIX: `input` in LLMTestCase is now the fully rendered prompt
        (template with {validation_case} and {similar_cases} filled in),
        so the judge LLM scores relevance against the actual query —
        not a raw template string with unfilled placeholders.
        """
        print("\n=== Starting Evaluation ===")

        # Verify Ollama is reachable before running expensive metrics
        print("Testing Ollama connection...")
        try:
            self.local_judge.generate("Hello")
            print("✓ Ollama connection successful")
        except Exception as exc:
            print(f"✗ Ollama connection failed: {exc}")
            return FAILED_CONNECTION_EVALUATION.copy()

        # FIX: `input` is now `rendered_prompt` — the exact query the LLM received
        test_case = LLMTestCase(
            input=original_content,
            actual_output=predicted_measures,
            expected_output=ground_truth,
            retrieval_context=retrieved_docs,
        )
        # test_case_for_retrive = LLMTestCase(
        #     input="เหตุการณ์ที่ต้องวิเคราะห์และกรณีที่คล้ายคลึงกัน 5 อันดับที่ใช้ในการอ้างอิง",
        #     actual_output=similar_cases_context,
        #     expected_output=original_content,
        #     retrieval_context=retrieved_docs,
        # )

        print("Running evaluation metrics...")
        try:
            self.relevancy_metric.measure(test_case)
            print(f"  AnswerRelevancy:       {self.relevancy_metric.score:.4f}")

            self.precision_metric.measure(test_case)
            print(f"  ContextualPrecision:   {self.precision_metric.score:.4f}")

            self.recall_metric.measure(test_case)
            print(f"  ContextualRecall:      {self.recall_metric.score:.4f}")

            self.contextual_relevancy_metric.measure(test_case)
            print(f"  ContextualRelevancy:   {self.contextual_relevancy_metric.score:.4f}")

        except Exception as exc:
            print(f"✗ Error during evaluation: {exc}")
            return {
                metric: {"score": "N/A", "reason": f"Error: {exc}"}
                for metric in ("AnswerRelevancy", "ContextualPrecision", "ContextualRecall", "ContextualRelevancy")
            }

        return {
            "AnswerRelevancy": {
                "score": self.relevancy_metric.score,
                "reason": self.relevancy_metric.reason,
            },
            "ContextualPrecision": {
                "score": self.precision_metric.score,
                "reason": self.precision_metric.reason,
            },
            "ContextualRecall": {
                "score": self.recall_metric.score,
                "reason": self.recall_metric.reason,
            },
            "ContextualRelevancy": {
                "score": self.contextual_relevancy_metric.score,
                "reason": self.contextual_relevancy_metric.reason,
            },
        }

    def write_to_csv(self, results_list: list[dict] | dict, csv_filename: str) -> None:
        """Write results to CSV. Accepts a single dict or a list of dicts."""
        if isinstance(results_list, dict):
            results_list = [results_list]

        # FIX: os.path.dirname("filename.csv") returns "" which causes makedirs to crash
        csv_dir = os.path.dirname(csv_filename)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        def get_metric_info(eval_dict: dict, metric_name: str) -> str:
            if eval_dict and metric_name in eval_dict:
                score = eval_dict[metric_name].get("score", "N/A")
                reason = eval_dict[metric_name].get(
                    "reason", eval_dict[metric_name].get("error", "No reason provided")
                )
                return f"{score}\n{reason}"
            return "N/A\nEvaluation not available"

        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["validate", "TopK", "results",
                 "AnswerRelevancy", "ContextualPrecision",
                 "ContextualRecall", "ContextualRelevancy"]
            )
            for result in results_list:
                eval_data = result.get("evaluation", {})
                writer.writerow([
                    result["val"],
                    result["TopK"],
                    result["results"],
                    get_metric_info(eval_data, "AnswerRelevancy"),
                    get_metric_info(eval_data, "ContextualPrecision"),
                    get_metric_info(eval_data, "ContextualRecall"),
                    get_metric_info(eval_data, "ContextualRelevancy"),
                ])

        print(f"✓ Results saved to {csv_filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    CHROMA_DB_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/DataPrep/res/database"
    FULL_DOCS_DIRECTORY = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/train/"
    VAL_FILE_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/valid_wo_measurement/2025-05-30 รายงานสืบสวนอุบัติเหตุเชิงลึก_RUTS-250101-08.docx"
    GT_FILE_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/valid_measurement_only/2025-05-30 รายงานสืบสวนอุบัติเหตุเชิงลึก_RUTS-250101-08.docx"
    VAL_FOLDER_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/valid_wo_measurement/"
    GT_FOLDER_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/valid_measurement_only/"
    K = 5
    PROCESS_MODE = "folder"  # "single" | "folder"

    pipeline = Pipeline(
        chroma_db_path=CHROMA_DB_PATH,
        full_docs_directory=FULL_DOCS_DIRECTORY,
        k=K,
        enable_evaluation=True,
    )

    if PROCESS_MODE == "single":
        print("=== Single File Processing ===")
        results = pipeline.predict(VAL_FILE_PATH, GT_FILE_PATH)
        print("✓ Single file processing completed")

    elif PROCESS_MODE == "folder":
        print("=== Folder Processing ===")
        results = pipeline.predict_folder(VAL_FOLDER_PATH, GT_FOLDER_PATH)
        print(f"✓ Folder processing completed — {len(results)} files processed")

    else:
        print("Invalid PROCESS_MODE. Use 'single' or 'folder'")
