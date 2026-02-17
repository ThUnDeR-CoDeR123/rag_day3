import os
from dotenv import load_dotenv
import requests
from typing import Optional

# Keep existing Gemini client as fallback
from google import genai

load_dotenv()


class RAGEngine:
    def __init__(self, retriever):
        self.retriever = retriever
        # Gemini client (fallback)
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # OpenRouter settings
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.2")

    def _call_openrouter(self, prompt: str) -> Optional[str]:
        if not self.openrouter_key:
            return None

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        rj = resp.json()

        # Try common response shapes
        # 1) choices -> message -> content (string or list)
        if isinstance(rj, dict) and "choices" in rj and len(rj["choices"]) > 0:
            choice = rj["choices"][0]
            # openrouter often returns choice['message']['content']
            msg = choice.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                return content
            if isinstance(content, list) and len(content) > 0:
                # content may be a list of dicts with 'text'
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
            # fallback to choice.text
            if "text" in choice and isinstance(choice["text"], str):
                return choice["text"]

        # 2) output style
        if isinstance(rj, dict) and "output" in rj:
            try:
                return rj["output"][0]["content"][0]["text"]
            except Exception:
                pass

        return None

    def generate_answer(self, query: str, top_k: int = 3):
        results = self.retriever.retrieve(query, top_k=top_k)

        context = "\n\n".join(
            [doc["content"] for doc in results]
        ) if results else ""

        if not context:
            return {"answer": "No relevant context found.", "sources": []}

        prompt = f"""
Use the following context to answer the question concisely.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

        # Prefer OpenRouter if key provided
        if self.openrouter_key:
            try:
                text = self._call_openrouter(prompt)
                if text:
                    return {"answer": text, "sources": [doc["metadata"] for doc in results]}
            except Exception:
                # fall through to Gemini fallback
                pass
        else:
            return {"answer": "No language model configured. Please set OPENROUTER_API_KEY.", "sources": [doc["metadata"] for doc in results], "context": context}
        # # Fallback to Gemini client
        # try:
        #     response = self.gemini_client.models.generate_content(
        #         model="gemini-3-flash-preview",
        #         contents=prompt,
        #         config=types.GenerateContentConfig(
        #             temperature=0.2,
        #             max_output_tokens=512,
        #         ),
        #     )
        #     text = getattr(response, "text", None) or (response.output[0].content[0].text if hasattr(response, "output") else None)
        # except Exception as e:
        #     text = f"An error occurred: {str(e)}"

        return {"answer": text, "sources": [doc["metadata"] for doc in results],"context": context}