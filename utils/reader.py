from docx import Document

class Reader:
    def read(self, file_path : str) -> str:
        try:
            document = Document(file_path)
            full_text = []
            for para in document.paragraphs:
                full_text.append(para.text)
            # Join all paragraph text with newline characters
            return '\n'.join(full_text)
        except Exception as error:
            return f"Error reading file: {error}"