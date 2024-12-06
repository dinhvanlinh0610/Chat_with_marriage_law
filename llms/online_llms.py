from groq import Groq

class OnlineLLM():
    def __init__(self, model_name = None, api_key = None):
        self.model_name = model_name
        self.model = None
        self.api_key = api_key
        self.llm = Groq(api_key=api_key)

        if self.model_name :
            self.model = Groq(api_key=api_key)

    def generate_content(self, prompt):

        if not self.model:
            raise ValueError("Model not found!")
        
        if self.model_name:
            response = self.model.chat.completions.create(
                model = self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là một chuyên gia pháp lý chuyên về Luật Hôn nhân và Gia đình Việt Nam. Tôi sẽ cung cấp câu hỏi và các đoạn văn bản tài liệu.
Nhiệm vụ của bạn:
1. Lọc các thông tin cần thiết từ tài liệu. 
2. Sử dụng nội dung đã lọc từ tài liệu được để trả lời câu hỏi.
2. Trích dẫn chính xác số điều, khoản, mục và nội dung luật từ tài liệu(tự bổ sung hoặc sửa chữa).
3. Nếu câu trả lời nằm ở nhiều điều khoản khác nhau, liệt kê tất cả, nhưng không mở rộng thêm bất kỳ ý kiến hay giải thích nào.
4. Nếu không tìm thấy câu trả lời trong tài liệu, chỉ trả lời: 'Không tìm thấy quy định phù hợp trong tài liệu được cung cấp.'"""
                    },
                    {
                        
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            try:
                content = response.choices[0].message.content
            except (IndexError, AttributeError):
                raise ValueError("Failed to parse")
        else:
            raise ValueError("Model not found!")
        
        if not isinstance(content, str):
            content = str(content)

        return content
    
    def generate_perfect_answer(self, prompt):
        if not self.model:
            raise ValueError("Model not found!")
        
        if self.model_name:
            response = self.model.chat.completions.create(
                model = self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là một chuyên gia Luật Hôn nhân và Gia đình Việt Nam. Nhận vào câu hỏi và câu trả lời dự đoán, đưa ra câu trả lời cuối cùng sau khi chỉnh sửa câu trả lời dự đoán cho chính xác cho câu hỏi với Fomart như sau \n
                        Câu hỏi : {Câu hỏi} \n
                        Câu trả lời cuối cùng: {Câu trả lời} \n
                        ."""
                    },
                    {
                        
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            try:
                content = response.choices[0].message.content
            except (IndexError, AttributeError):
                raise ValueError("Failed to parse")
        else:
            raise ValueError("Model not found!")
        
        if not isinstance(content, str):
            content = str(content)

        return content
    
    def generate_content_answer(self, prompt):
        if not self.model:
            raise ValueError("Model not found!")
        
        if self.model_name:
            response = self.model.chat.completions.create(
                model = self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là một chuyên gia Luật Hôn nhân và Gia đình Việt Nam. Nhận vào 1 câu hỏi và các tài liệu liên quan, lọc vứt bỏ các phần không liên quan và giữ lại các phân đoạn chính xác, bổ sung nếu thấy tài liệu chưa đầy đủ nội dung luật. \n
                        Định dạng trả lời của bạn như sau: 
                        "Thông tin tài liệu: Điều , Mục, Khoản, Nội dung luật.
                        Ví dụ: Điều 9 Luật hôn nhân và Gia đình 2014, Các hành vi cấm bao gồm :
                        Kêt hôn giả tạo, ly hôn giả tạo (khoản a)
                        Tảo hôn, cưỡng ép kết hôn, lừa dối kết hôn , cản trở kết hôn ( khoản b)
                        ...)"
                        """
                    },
                    {
                        
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            try:
                content = response.choices[0].message.content
            except (IndexError, AttributeError):
                raise ValueError("Failed to parse")
        else:
            raise ValueError("Model not found!")
        
        if not isinstance(content, str):
            content = str(content)

        return content
    






    
    
