"""
LLM narrator module for generating explainable summaries from structured outputs.
This module is optional and disabled unless environment flags are set.
"""

import json
import os
import ast
import re
from typing import Any, Dict, Optional

import requests


class NarrationError(Exception):
    """Raised when narration cannot be generated."""


class GeminiNarrator:
    """Generate clinician and patient-friendly narratives from structured findings."""

    def __init__(self):
        self.enabled = os.environ.get("LLM_ENABLED", "false").lower() == "true"
        self.provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
        # Backward-compatible aliases for deprecated Gemini model names.
        if self.model in {
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-2.0-flash",
        }:
            self.model = "gemini-2.5-flash"
        self.fallback_models = ["gemini-2.5-flash", "gemini-flash-latest"]
        self.timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", "25"))

    def is_available(self) -> bool:
        return self.enabled and self.provider == "gemini" and bool(self.api_key)

    def generate_narrative(self, report_payload: Dict[str, Any], style: str = "both") -> Dict[str, Any]:
        """Generate a strict JSON narrative response based only on given payload."""
        if not self.is_available():
            raise NarrationError("LLM narration is disabled or Gemini API key is missing")

        prompt = self._build_prompt(report_payload, style)
        data = self._request_with_fallback(prompt, json_mode=True)
        text = self._extract_text(data)
        if not text:
            raise NarrationError("Gemini returned an empty response")

        parsed = self._safe_parse_json(text)
        if not parsed:
            # Graceful fallback: use raw text as summary
            fallback_text = text.strip()
            # Clean up if it looks like partial JSON
            if fallback_text.startswith("{"):
                for key in ["clinician_summary", "patient_summary"]:
                    match = re.search(rf'"{key}"\s*:\s*"([^"]*)', fallback_text)
                    if match:
                        fallback_text = match.group(1)
                        break
                else:
                    # Can't extract, use a portion of the raw text
                    fallback_text = re.sub(r'[{}":]', '', fallback_text)[:500] if fallback_text else "Summary generation failed."
            parsed = {
                "clinician_summary": fallback_text,
                "patient_summary": fallback_text,
                "safety_note": "AI-generated summary for reference only.",
            }

        return self._normalize_output(parsed)

    def chat(self, context: Dict[str, Any], history: list, message: str) -> str:
        """Chat with the LLM about the report context."""
        if not self.is_available():
            raise NarrationError("LLM narration is disabled or Gemini API key is missing")

        # Build prompt with context and history
        system_instruction = (
            "You are a helpful medical assistant. You are answering questions about a specific medical report.\n"
            "Here is the report context:\n"
            f"{json.dumps(context, indent=2, default=str)}\n\n"
            "Rules:\n"
            "1. Answer strictly based on the provided report context.\n"
            "2. If the user asks about something not in the report, say it's not available in the report.\n"
            "3. Provide clear, empathetic, and accurate explanations for patients.\n"
            "4. Do not provide medical advice or diagnosis. Always recommend consulting a doctor.\n"
            "5. Keep answers concise.\n"
        )
        
        # In a real chat app, we would send the full history to the API 'contents'.
        # However, for simplicity using the REST API without a stateful session object,
        # we can construct a single prompt with history or use the 'contents' array if we want multi-turn.
        # Gemini API 'contents' supports multi-turn.
        
        contents = []
        
        # Add system instruction as the first user part (since system instructions are separate in v1beta but we can also just prepend context)
        # Using the simple approach of appending context to the first message or system instructions.
        # For 'generateContent', we pass a list of contents.
        
        # We will flatten context into the first message or treat it as a "pre-prompt"
        
        # Current simplified history approach:
        # history is a list of {"role": "user" | "model", "text": "..."}
        
        contents.append({
            "role": "user",
            "parts": [{"text": system_instruction}]
        })
        
        contents.append({
            "role": "model",
            "parts": [{"text": "I understand. I am ready to answer questions about this report."}]
        })
        
        for msg in history:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.get("text", "")}]
            })
            
        # Add the new message
        contents.append({
            "role": "user",
            "parts": [{"text": message}]
        })
        
        data = self._request_with_fallback(contents, json_mode=False)
        return self._extract_text(data)

    def _request_with_fallback(self, content_payload: Any, json_mode: bool = True) -> Dict[str, Any]:
        # content_payload can be a string (single prompt) or a list (chat history)
        if isinstance(content_payload, str):
            contents = [{
                "role": "user",
                "parts": [{"text": content_payload}],
            }]
        else:
            contents = content_payload

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096,
            },
        }
        
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        tried = []
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]

        last_error = None
        for model_name in models_to_try:
            tried.append(model_name)
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={self.api_key}"
            )
            try:
                response = requests.post(url, json=body, timeout=self.timeout_seconds)
                if response.status_code == 404:
                    last_error = f"Model unavailable: {model_name}"
                    continue
                response.raise_for_status()
                # Use the first successful model for subsequent calls.
                self.model = model_name
                return response.json()
            except requests.RequestException as exc:
                last_error = str(exc)
                # Retry only for model unavailability; propagate other failures.
                if getattr(exc.response, 'status_code', None) == 404:
                    continue
                raise NarrationError(f"Gemini request failed: {exc}") from exc

        raise NarrationError(
            "Gemini request failed: no compatible model found "
            f"(tried: {', '.join(tried)}; last_error: {last_error})"
        )

    def _extract_text(self, response_data: Dict[str, Any]) -> str:
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
            return parts[0].get("text", "").strip()
        except Exception:
            return ""

    def _safe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        # Attempt strict JSON parsing first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Handle markdown code blocks such as ```json ... ```.
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: extract the first JSON object block if wrapped with extra text.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Last fallback for Python-like dict output.
        try:
            py_obj = ast.literal_eval(candidate)
            if isinstance(py_obj, dict):
                return py_obj
        except (ValueError, SyntaxError):
            return None

        return None

    def _normalize_output(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "clinician_summary": parsed.get("clinician_summary", ""),
            "patient_summary": parsed.get("patient_summary", ""),
            "bullet_findings": parsed.get("bullet_findings", []),
            "data_used": parsed.get("data_used", []),
            "limitations": parsed.get("limitations", []),
            "safety_note": parsed.get(
                "safety_note",
                "This output is decision support only and not a medical diagnosis.",
            ),
        }

    def _build_prompt(self, report_payload: Dict[str, Any], style: str) -> str:
        style = style.lower().strip()
        if style not in {"clinician", "patient", "both"}:
            style = "both"

        instructions = {
            "clinician": "Generate only clinician_summary with technical language.",
            "patient": "Generate only patient_summary with simple language.",
            "both": "Generate both clinician_summary and patient_summary.",
        }

        schema = {
            "clinician_summary": "string",
            "patient_summary": "string",
            "bullet_findings": ["string"],
            "safety_note": "string",
        }

        return (
            "You are a clinical writing assistant.\n"
            "Use ONLY the structured JSON provided below.\n"
            "Do NOT invent values, diagnoses, causes, or treatment plans.\n"
            "If data is missing, say 'Not available'.\n"
            "Never alter numeric values.\n"
            f"{instructions[style]}\n"
            "Return strictly valid JSON with this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Rules:\n"
            "1) bullet_findings must include exact metric values where present.\n"
            "2) Keep summary faithful to inputs.\n\n"
            "Input JSON:\n"
            f"{json.dumps(report_payload, indent=2, default=str)}"
        )
