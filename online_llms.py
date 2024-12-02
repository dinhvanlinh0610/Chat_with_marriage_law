from groq import Groq

class OnlineLLM():
    def __init__(self, model_name = "llama3-8b-8192", api_key = None):
        self.model_name = model_name
        self.model = None
        self.api_key = api_key
        self.llm = Groq(api_key=api_key)

        if self.model_name == "llama3-8b-8192":
            self.model = Groq(api_key=api_key)

    def generate_content(self, prompt):

        if not self.model:
            raise ValueError("Model not found!")
        
        if self.model_name == "llama3-8b-8192":
            response = self.model.chat.completions.create(
                model = self.model_name,
                messages=[
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





    
    
