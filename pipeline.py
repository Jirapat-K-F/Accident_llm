import csv
import chromadb
from docx import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import pandas as pd
import config
from utils import Reader, Indexing
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

# Set DeepEval timeout configuration
os.environ['DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE'] = '300'  # 5 minutes timeout
os.environ['DEEPEVAL_TOTAL_TIMEOUT_SECONDS_OVERRIDE'] = '600'  # 10 minutes total timeout

class Pipeline:
    def __init__(self, chroma_db_path, full_docs_directory, k=5, model_name="llama3", collection_name="accident_report_summaries", enable_evaluation=True):
        self.chroma_db_path = chroma_db_path
        self.full_docs_directory = full_docs_directory
        self.k = k
        self.enable_evaluation = enable_evaluation
        self.reader = Reader()
        self.indexing = Indexing(model_name=model_name)
        self.llm = OllamaLLM(model=model_name)
        
        # Only initialize local judge if evaluation is enabled
        if self.enable_evaluation:
            # Try llama3.1 first as it might be more compatible with structured output
            try:
                self.local_judge = OllamaModel(model="llama3.1:latest", base_url="http://localhost:11434")
            except:
                # Fallback to deepseek if llama3.1 fails
                self.local_judge = OllamaModel(model="deepseek-r1:8B", base_url="http://localhost:11434")
        else:
            self.local_judge = None

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_collection(collection_name)

        self.template = """คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์อุบัติเหตุและการเสนอมาตรการแก้ไขปัญหา

        กรุณาเสนอมาตรการแก้ไขปัญหาระยะสั้นที่เหมาะสมต่อเหตุการณ์ในรายงานสืบสวนอุบัติเหตุเชิงลึกต่อไปนี้ โดยอ้างอิงจากกรณีที่คล้ายคลึงกันและมาตรการที่ใช้ในกรณีเหล่านั้น:

        === เหตุการณ์ที่ต้องวิเคราะห์ ===
        {validation_case}

        === กรณีที่คล้ายคลึงและมาตรการที่ใช้ (สำหรับอ้างอิง) ===
        {similar_cases}

        === คำแนะนำ ===
        อ้างอิงจากกรณีที่คล้ายคลึงเพื่อเสนอมาตรการที่เหมาะสม
        ปรับมาตรการให้เหมาะกับลักษณะเฉพาะของกรณีนี้
        บอกแค่มาตรการระยะสั้นเท่านั้น
        ระบุผู้รับผิดชอบสำหรับแต่ละมาตรการ
        หากไม่สามารถหาแนวทางแก้ไขได้จากกรณีที่คล้ายคลึง สามารถตอบสั้นๆ ได้ว่า "ไม่พบมาตรการที่เหมาะสม"
        กรุณาให้ผลลัพธ์เป็นมาตรการที่ชัดเจน เป็นระบบ และสามารถนำไปปฏิบัติได้จริง แสดงผลลัพธ์แค่มาตรการที่แนะนำเท่านั้นโดยไม่ต้องใส่ข้อมูลอื่นเพิ่มเติม:
        """
    
    def find_top_k(self, val_summary):
        print(f"Finding {self.k} similar files from ChromaDB...")
        
        # Query similar documents
        results = self.collection.query(
            query_texts=[val_summary],
            n_results=self.k
        )
        
        # Extract results
        distances = results['distances'][0]
        ids_result = results['ids'][0]
        
        similar_files = []
        for rank, (doc_id, dist) in enumerate(zip(ids_result, distances), start=1):
            sim_score = 1 - dist  # Convert distance to similarity
            similar_files.append({
                "rank": rank,
                "filename": doc_id,
                "similarity_score": round(sim_score, 4)
            })
            print(f"  {rank}. {doc_id} (similarity: {round(sim_score, 4)})")
        
        print(f"✓ Found {len(similar_files)} similar files")
        return similar_files
    
    def get_full_documents(self, similar_files):
        print(f"Retrieving full documents from directory...")
        
        full_documents = {}
        
        for file_info in similar_files:
            filename = file_info["filename"]
            file_path = os.path.join(self.full_docs_directory, filename)
            
            # Try to read the file
            if os.path.exists(file_path):
                try:
                    content = self.reader.read(file_path)
                    if not content.startswith("Error reading file"):
                        full_documents[filename] = content
                        print(f"  ✓ Retrieved: {filename}")
                    else:
                        print(f"  ✗ Failed to read: {filename}")
                except Exception as e:
                    print(f"  ✗ Error reading {filename}: {e}")
            else:
                print(f"  ✗ File not found: {file_path}")
        
        print(f"✓ Retrieved {len(full_documents)} full documents")
        return full_documents
    
    def get_prediction(self, val_content, similar_cases_context):
        
        
        prompt = ChatPromptTemplate.from_template(self.template)
        chain = prompt | self.llm
        return chain.invoke({
            "validation_case": val_content,
            "similar_cases": similar_cases_context
        })   

    def predict(self, val_file_path : str, gt_file_path : str) -> str:
        results = {}
        val_content = self.reader.read(val_file_path)
        val_summary = self.indexing.summarize_with_llm(val_content)
        print("Validation Content Loaded.")
        
        similar_files = self.find_top_k(val_summary)
        full_documents = self.get_full_documents(similar_files)
        
        # Combine similar cases content
        similar_cases_context = "\n\n".join(full_documents.values())
        
        prediction = self.get_prediction(val_content, similar_cases_context)

        gt = self.reader.read(gt_file_path)
        print(gt)
        # Run evaluation only if enabled
        if self.enable_evaluation and self.local_judge is not None:
            evaluation_results = self.evaluate(prediction, gt, list(full_documents.values()))
        else:
            print("Evaluation disabled or local judge not available. Skipping evaluation...")
            evaluation_results = {
                "AnswerRelevancy": {"score": "Disabled", "reason": "Evaluation was disabled"},
                "ContextualPrecision": {"score": "Disabled", "reason": "Evaluation was disabled"},
                "ContextualRecall": {"score": "Disabled", "reason": "Evaluation was disabled"}, 
                "ContextualRelevancy": {"score": "Disabled", "reason": "Evaluation was disabled"}
            }

        results['val'] = os.path.basename(val_file_path)
        results['TopK'] = '\n'.join([f"{file['rank']}. {file['filename']} (similarity: {file['similarity_score']})" for file in similar_files])
        results['results'] = prediction
        self.write_to_csv(results, os.path.join("res/eval", f"validate_results.csv"), eval=evaluation_results)
        return results

    def write_to_csv(self, results: dict, csv_filename: str, eval: dict = None):
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["validate", "TopK", "results", "AnswerRelevancy", "ContextualPrecision", "ContextualRecall", "ContextualRelevancy"])
            
            # Helper function to safely get score and reason
            def get_metric_info(metric_name):
                if eval and metric_name in eval:
                    score = eval[metric_name].get("score", "N/A")
                    reason = eval[metric_name].get("reason", eval[metric_name].get("error", "No reason provided"))
                    return f"{score}\n{reason}"
                return "N/A\nEvaluation not available"
            
            writer.writerow([
                results["val"], results["TopK"], results["results"], 
                get_metric_info("AnswerRelevancy"),
                get_metric_info("ContextualPrecision"),
                get_metric_info("ContextualRecall"),
                get_metric_info("ContextualRelevancy")
                ])
        print(f"Results saved to {csv_filename}")

    def evaluate(self, predicted_measures: str, val_content: str, top_k: list[str]) -> dict:
        print("\n=== Starting Evaluation ===")
        
        # Test Ollama connection first
        print("Testing Ollama connection...")
        try:
            # Simple test generation
            test_response = self.local_judge.generate("Hello")
            print("✓ Ollama connection successful")
        except Exception as e:
            print(f"✗ Ollama connection failed: {e}")
            print("Please ensure Ollama is running and the deepseek-r1:8B model is available")
            return {
                "AnswerRelevancy": {"score": "N/A", "error": "Ollama connection failed"},
                "ContextualPrecision": {"score": "N/A", "error": "Ollama connection failed"},
                "ContextualRecall": {"score": "N/A", "error": "Ollama connection failed"},
                "ContextualRelevancy": {"score": "N/A", "error": "Ollama connection failed"}
            }

        test_case = LLMTestCase(
            input=self.template,
            actual_output=predicted_measures,
            retrieval_context=top_k,
            expected_output=val_content
        )

        # 3. Define the Metric using the local judge
        print("Initializing evaluation metrics...")
        
        relevancy_metric = AnswerRelevancyMetric(
            threshold=0.8,
            model=self.local_judge  # This forces DeepEval to use Ollama instead of OpenAI
        )

        presision_metric = ContextualPrecisionMetric(
            threshold=0.8,
            model=self.local_judge
        )

        recall_metric = ContextualRecallMetric(
            threshold=0.8,
            model=self.local_judge
        )

        contextual_relevancy_metric = ContextualRelevancyMetric(
            threshold=0.8,
            model=self.local_judge
        )
        
        print("✓ Evaluation metrics initialized successfully")

        # 4. Run the evaluation
        try:
            print("Running AnswerRelevancy evaluation...")
            relevancy_metric.measure(test_case)
            print("Running ContextualPrecision evaluation...")
            presision_metric.measure(test_case)
            print("Running ContextualRecall evaluation...")
            recall_metric.measure(test_case)
            print("Running ContextualRelevancy evaluation...")
            contextual_relevancy_metric.measure(test_case)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Continuing without evaluation scores...")
            return {
                "AnswerRelevancy": {"score": "N/A", "reason": f"Error: {str(e)}"},
                "ContextualPrecision": {"score": "N/A", "reason": f"Error: {str(e)}"},
                "ContextualRecall": {"score": "N/A", "reason": f"Error: {str(e)}"},
                "ContextualRelevancy": {"score": "N/A", "reason": f"Error: {str(e)}"}
            }

        print(f"AnswerRelevancy Score: {relevancy_metric.score}")
        print(f"Reason: {relevancy_metric.reason}")
        print(f"ContextualPrecision Score: {presision_metric.score}")
        print(f"Reason: {presision_metric.reason}")
        print(f"ContextualRecall Score: {recall_metric.score}")
        print(f"Reason: {recall_metric.reason}")
        print(f"ContextualRelevancy Score: {contextual_relevancy_metric.score}")
        print(f"Reason: {contextual_relevancy_metric.reason}")


        return {            
            "AnswerRelevancy": {
                "score": relevancy_metric.score,
                "reason": relevancy_metric.reason
                },
            "ContextualPrecision": {
                "score": presision_metric.score,
                "reason": presision_metric.reason
                },
            "ContextualRecall": {
                "score": recall_metric.score,
                "reason": recall_metric.reason
                },
            "ContextualRelevancy": {
                "score": contextual_relevancy_metric.score,
                "reason": contextual_relevancy_metric.reason
                }
        }




# Example usage
if __name__ == "__main__":
    # Configuration
    CHROMA_DB_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/DataPrep/res/database"
    # CHROMA_DB_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/DataPrep/res/summary_data/sum_llama3/chroma_db"
    FULL_DOCS_DIRECTORY = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/train/"
    VAL_FILE_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/valid_wo_measurement/2025-05-30 รายงานสืบสวนอุบัติเหตุเชิงลึก_RUTS-250101-08.docx"
    # VAL_FILE_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/valid_wo_measurement/2025-06-28 รายงานสืบสวนอุบัติเหตุเชิงลึก_RUTS-250519-13.docx"
    # VAL_FILE_PATH = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/valid_wo_measurement/2025-04-16 รายงานสืบสวนอุบัติเหตุเชิงลึก_NO-250101-01.docx"
    GT_file_path = "C:/Users/User/Desktop/chula/Year3/indiv_ai_agent/dataset/valid_measurement_only/2025-05-30 รายงานสืบสวนอุบัติเหตุเชิงลึก_RUTS-250101-08.docx"
    K = 5  # Number of similar documents to retrieve
    
    # Initialize pipeline
    pipeline = Pipeline(
        chroma_db_path=CHROMA_DB_PATH,
        full_docs_directory=FULL_DOCS_DIRECTORY,
        k=K,
        enable_evaluation=True  # Set to False to skip evaluation if you encounter timeout issues
        # collection_name="summary_comparison"
    )
    
    # Run pipeline
    results = pipeline.predict(VAL_FILE_PATH, GT_file_path)
    # print("Prediction Results:")
    # print(results)

