from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class Request(BaseModel):
    task: str
    document_text: str = ""
    selection_text: str = ""

@app.post("/generate")
def generate(req: Request):
    if req.task == "executive_summary":
        prompt = f"Summarise the following:\n\n{req.document_text}"
    elif req.task == "style_enhance":
        prompt = f"Improve style without changing meaning:\n\n{req.selection_text}"
    else:
        return {"error": "Unsupported task"}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return {"result": response.choices[0].message.content}
