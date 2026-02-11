from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class Indexing:
    def __init__(self, model_name="llama3"):
        self.llm = OllamaLLM(model=model_name)

    def summarize_with_llm(self, text_content: str) -> str:
        # Prompt
        template = """สรุปเหตุการณ์ที่เกิดขึ้นต่อไปนี้ออกมาเป็นหัวข้ออย่างละเอียด โดยมีทั้งข้อมูลเหตุการณ์และรายละเอียด ปัจจัย ให้ได้ความยาวอย่างน้อย 1000 คำเป็นภาษาไทย โดยไม่ใส่มาตรการแก้ไขปัญหา:
        {context}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | self.llm

        return chain.invoke({"context": text_content})