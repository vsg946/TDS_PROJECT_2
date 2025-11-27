import os
import re
import time
import json
import base64
import tempfile
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright, Page

# AI Pipe token from environment
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")


class EnhancedQuizSolver:
    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.session = requests.Session()
        self.visited = set()
        self.cache: Dict[str, Tuple[str, bytes, str]] = {}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _debug(self, *args):
        print("[DEBUG]", *args, flush=True)

    def _find_submit_url(self, html: str, page_url: str) -> Optional[str]:
        """Find /submit endpoint with multiple strategies."""
        patterns = [
            r"https?://[^\s\"'<>]+/submit",
            r"[\"']([^\"']*submit[^\"']*)[\"']",
            r"action=[\"']([^\"']*)[\"']",
        ]

        for pattern in patterns:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                url = m.group(1) if m.groups() else m.group(0)
                if url.startswith("http"):
                    return url
                return urljoin(page_url, url)

        # Look in forms
        soup = BeautifulSoup(html, "html.parser")
        for form in soup.find_all("form"):
            action = form.get("action", "")
            if "submit" in action.lower():
                return urljoin(page_url, action)

        # Default fallback
        if "/submit" in html.lower():
            p = urlparse(page_url)
            return f"{p.scheme}://{p.netloc}/submit"

        return None

    def _decode_base64(self, s: str) -> Optional[str]:
        """Decode base64 with URL-safe variant support."""
        try:
            s = s.replace("-", "+").replace("_", "/")
            pad = len(s) % 4
            if pad:
                s += "=" * (4 - pad)
            return base64.b64decode(s).decode("utf-8", "ignore")
        except Exception:
            return None

    def _extract_hidden_data(self, html: str, page: Page) -> Dict[str, Any]:
        """Extract hidden data: atob, JSON, hidden inputs, data attrs, etc."""
        hidden_data: Dict[str, Any] = {
            "atob_decoded": [],
            "json_objects": [],
            "hidden_inputs": [],
            "data_attributes": [],
            "script_variables": [],
            "comments": [],
            "meta_tags": [],
            "localStorage": {},
            "cookies": {},
        }

        soup = BeautifulSoup(html, "html.parser")

        # 1. atob() payloads
        atob_pattern = r'atob\(\s*[`"\']([^`"\']+)[`"\']\s*\)'
        for m in re.finditer(atob_pattern, html):
            decoded = self._decode_base64(m.group(1))
            if decoded:
                hidden_data["atob_decoded"].append(decoded)

        # 2. JSON and variables from scripts
        for script in soup.find_all("script"):
            script_text = script.string or ""

            # JSON-ish objects
            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            for match in re.finditer(json_pattern, script_text):
                try:
                    obj = json.loads(match.group(0))
                    hidden_data["json_objects"].append(obj)
                except Exception:
                    pass

            # var / let / const
            var_pattern = r"(?:var|let|const)\s+(\w+)\s*=\s*([^;]+);"
            for match in re.finditer(var_pattern, script_text):
                var_name = match.group(1)
                var_value = match.group(2).strip()
                hidden_data["script_variables"].append(
                    {"name": var_name, "value": var_value}
                )

        # 3. Hidden inputs
        for inp in soup.find_all("input", type="hidden"):
            hidden_data["hidden_inputs"].append(
                {"name": inp.get("name", ""), "value": inp.get("value", "")}
            )

        # 4. Data attributes
        for elem in soup.find_all(
            attrs=lambda x: x and any(k.startswith("data-") for k in x)
        ):
            data_attrs = {
                k: v for k, v in elem.attrs.items() if isinstance(k, str) and k.startswith("data-")
            }
            if data_attrs:
                hidden_data["data_attributes"].append(
                    {"tag": elem.name, "attributes": data_attrs}
                )

        # 5. Meta tags
        for meta in soup.find_all("meta"):
            if meta.get("name") or meta.get("property"):
                hidden_data["meta_tags"].append(
                    {
                        "name": meta.get("name") or meta.get("property"),
                        "content": meta.get("content", ""),
                    }
                )

        # 6. localStorage
        try:
            storage = page.evaluate("() => JSON.stringify(localStorage)")
            hidden_data["localStorage"] = json.loads(storage)
        except Exception:
            pass

        # 7. Cookies
        try:
            cookies = page.context.cookies()
            hidden_data["cookies"] = {c["name"]: c["value"] for c in cookies}
        except Exception:
            pass

        return hidden_data

    # ------------------------------------------------------------------
    # Download + File Summarisation
    # ------------------------------------------------------------------
    def _download_file(self, href: str, base_url: str) -> Optional[Tuple[str, bytes, str]]:
        if not href:
            return None

        if not href.lower().startswith("http"):
            href = urljoin(base_url, href)

        cache_key = hashlib.md5(href.encode()).hexdigest()
        if cache_key in self.cache:
            self._debug("Using cached file:", href)
            return self.cache[cache_key]

        self._debug("Downloading file:", href)
        try:
            r = self.session.get(href, timeout=30)
            if r.status_code != 200:
                self._debug("Download status", r.status_code)
                return None
            content = r.content
        except Exception as e:
            self._debug("Download error:", e)
            return None

        ct = r.headers.get("content-type", "").lower()
        ext = ""
        ftype = "other"
        lower_href = href.lower()

        if "pdf" in ct or lower_href.endswith(".pdf"):
            ext = ".pdf"
            ftype = "pdf"
        elif "csv" in ct or lower_href.endswith(".csv"):
            ext = ".csv"
            ftype = "csv"
        elif "excel" in ct or any(lower_href.endswith(x) for x in [".xlsx", ".xls"]):
            ext = ".xlsx"
            ftype = "xlsx"
        elif "json" in ct or lower_href.endswith(".json"):
            ext = ".json"
            ftype = "json"
        elif any(lower_href.endswith(x) for x in [".mp3", ".wav", ".ogg", ".m4a", ".flac"]):
            ext = "." + lower_href.split(".")[-1].split("?")[0]
            ftype = "audio"
        elif any(lower_href.endswith(x) for x in [".txt", ".log"]):
            ext = ".txt"
            ftype = "text"
        else:
            name = href.split("/")[-1]
            if "." in name:
                ext = "." + name.split(".")[-1].split("?")[0]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(content)
        tmp.close()

        result = (tmp.name, content, ftype)
        self.cache[cache_key] = result
        return result

    def _summarise_csv_xlsx(self, path: str) -> Dict[str, Any]:
        """Analyse CSV/XLSX: stats, preview, value counts."""
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, nrows=5000)
            else:
                df = pd.read_excel(path, nrows=5000)
        except Exception as e:
            self._debug("CSV/XLSX read error:", e)
            return {"type": "table", "error": str(e)}

        summary: Dict[str, Any] = {
            "type": "table",
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "null_counts": df.isnull().sum().to_dict(),
        }

        # Numeric stats
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            stats: Dict[str, Any] = {}
            for col in numeric_cols:
                col_ser = df[col]
                if col_ser.isna().all():
                    stats[col] = {
                        "mean": None,
                        "median": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "sum": None,
                        "count": 0,
                    }
                else:
                    stats[col] = {
                        "mean": float(col_ser.mean()),
                        "median": float(col_ser.median()),
                        "std": float(col_ser.std()),
                        "min": float(col_ser.min()),
                        "max": float(col_ser.max()),
                        "sum": float(col_ser.sum()),
                        "count": int(col_ser.count()),
                    }
            summary["numeric_stats"] = stats

        # Preview rows
        preview = df.head(50)
        summary["sample_rows"] = preview.to_dict(orient="records")

        # Value counts for small-cardinality categoricals
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        value_counts: Dict[str, Dict[str, int]] = {}
        for col in categorical_cols:
            if df[col].nunique() <= 20:
                value_counts[col] = df[col].value_counts().to_dict()
        if value_counts:
            summary["value_counts"] = value_counts

        return summary

    def _summarise_pdf(self, path: str) -> Dict[str, Any]:
        """Summarise PDFs: text, tables, numbers."""
        out: Dict[str, Any] = {"type": "pdf", "pages": [], "metadata": {}}
        try:
            with pdfplumber.open(path) as pdf:
                out["metadata"] = pdf.metadata or {}
                max_pages = min(len(pdf.pages), 10)
                for i in range(max_pages):
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    page_data: Dict[str, Any] = {
                        "page_index": i,
                        "text": text[:6000],
                        "tables": [],
                        "numbers_found": [],
                    }

                    tables = page.extract_tables()
                    if tables:
                        for t in tables[:5]:
                            if t:
                                page_data["tables"].append(t[:20])

                    numbers = re.findall(r"-?\d+\.?\d*", text)
                    page_data["numbers_found"] = numbers[:100]

                    out["pages"].append(page_data)
        except Exception as e:
            self._debug("PDF summary error:", e)
            out["error"] = str(e)
        return out

    def _summarise_text_file(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(10000)
            return {
                "type": "text",
                "content": content,
                "numbers": re.findall(r"-?\d+\.?\d*", content),
            }
        except Exception as e:
            return {"type": "text", "error": str(e)}

    def _summarise_json_file(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._debug("JSON file read error:", e)
            return {"type": "json", "error": str(e)}

        summary: Dict[str, Any] = {"type": "json", "data": data}
        if isinstance(data, dict):
            summary["keys"] = list(data.keys())
            summary["structure"] = "object"
        elif isinstance(data, list):
            summary["length"] = len(data)
            summary["structure"] = "array"
            if data and isinstance(data[0], dict):
                summary["sample_keys"] = list(data[0].keys())
        return summary

    # ------------------------------------------------------------------
    # Audio handling + CSV instruction execution
    # ------------------------------------------------------------------
    def _transcribe_audio_llm(self, audio_bytes: bytes, ext: str) -> Optional[str]:
        if not AIPIPE_TOKEN:
            self._debug("No AIPIPE_TOKEN set; cannot transcribe audio")
            return None

        self._debug(f"Transcribing audio via AIPipe ({len(audio_bytes)} bytes)...")
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        fmt = ext.lstrip(".") or "mp3"

        url = "https://aipipe.org/openrouter/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a perfect speech-to-text engine. "
                        "Transcribe the audio exactly, including numbers and instructions. "
                        "Return ONLY the transcript text."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": fmt},
                        }
                    ],
                },
            ],
        }

        try:
            r = self.session.post(url, headers=headers, json=payload, timeout=90)
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                transcript = " ".join(text_parts).strip()
            else:
                transcript = str(content).strip()
            self._debug("Audio transcript:", transcript[:300])
            return transcript
        except Exception as e:
            self._debug("AIPipe audio error:", e)
            return None

    def _try_audio_csv_solve(self, transcript: str, csv_path: str) -> Optional[float]:
        """
        Handle instructions like:
        - "Add all values in column 1 greater than the cutoff 26779 in demo-audio-data.csv"
        """
        t = transcript.lower()
        if not any(word in t for word in ["sum", "add", "total", "cutoff", "greater than"]):
            return None

        # Find cutoff
        cutoff = None
        m = re.search(r"cutoff[^0-9]*([0-9]+)", t)
        if not m:
            m = re.search(r"greater than[^0-9]*([0-9]+)", t)
        if m:
            cutoff = float(m.group(1))

        # Column index
        col_index = 0  # default first column
        m = re.search(r"column\s+([0-9]+)", t)
        if m:
            col_index = max(0, int(m.group(1)) - 1)
        elif "first column" in t or "column one" in t:
            col_index = 0

        self._debug("Audio+CSV parsed: cutoff=", cutoff, "col_index=", col_index)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self._debug("Audio-CSV read error:", e)
            return None

        # Try to pick numeric column at given index
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            if col_index < len(numeric_cols):
                col_name = numeric_cols[col_index]
            else:
                col_name = numeric_cols[0]
        else:
            # fall back to positional
            if col_index >= df.shape[1]:
                col_index = 0
            col_name = df.columns[col_index]

        ser = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if cutoff is not None:
            ser = ser[ser > cutoff]

        total = float(ser.sum())
        self._debug("Audio-CSV computed sum:", total)
        return total

    # ------------------------------------------------------------------
    # LLM reasoning
    # ------------------------------------------------------------------
    def _llm_solve_from_context(self, context: Dict[str, Any]) -> Any:
        if not AIPIPE_TOKEN:
            self._debug("No AIPIPE_TOKEN set; cannot use LLM for reasoning")
            return None

        url = "https://aipipe.org/openrouter/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        }

        context_text = self._build_context_text(context)

        system_prompt = '''You are an expert data analyst solving quiz questions. You will receive:
1. Page content (text, HTML)
2. Hidden data (decoded scripts, JSON, data attributes)
3. File summaries (CSV statistics, PDF content, JSON data)
4. Audio transcripts with instructions

Your task:
- Read ALL information carefully
- Identify the question being asked
- Locate relevant data (cutoff values, column names, conditions)
- Perform calculations if needed (sums, counts, filtering)
- Determine the FINAL ANSWER

CRITICAL: Return ONLY valid JSON in this exact format:
{"answer": <your_answer>}

Examples:
- {"answer": 42}
- {"answer": "text response"}
- {"answer": 3.14159}
- {"answer": true}

Do NOT include explanations or extra text.'''

        payload = {
            "model": "openai/gpt-4.1-nano",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Quiz context:\n\n{context_text}"},
            ],
        }

        try:
            r = self.session.post(url, headers=headers, json=payload, timeout=120)
            data = r.json()
            content = data["choices"][0]["message"]["content"]

            if isinstance(content, str):
                answer_obj = json.loads(content)
            else:
                answer_obj = content

            self._debug("LLM answer object:", answer_obj)
            if isinstance(answer_obj, dict) and "answer" in answer_obj:
                return answer_obj["answer"]
        except Exception as e:
            self._debug("LLM reasoning error:", e)

        return None

    def _build_context_text(self, context: Dict[str, Any]) -> str:
        parts: List[str] = []

        parts.append("=== PAGE INFORMATION ===")
        parts.append(f"URL: {context.get('page_url', '')}")
        parts.append(f"\nVISIBLE TEXT:\n{context.get('page_text', '')[:8000]}")

        if context.get("hidden_data"):
            hd = context["hidden_data"]
            parts.append("\n=== HIDDEN DATA ===")
            if hd.get("atob_decoded"):
                parts.append(
                    f"\nBase64 Decoded Content:\n{json.dumps(hd['atob_decoded'], indent=2)}"
                )
            if hd.get("script_variables"):
                parts.append(
                    f"\nJavaScript Variables:\n{json.dumps(hd['script_variables'][:20], indent=2)}"
                )
            if hd.get("hidden_inputs"):
                parts.append(
                    f"\nHidden Form Inputs:\n{json.dumps(hd['hidden_inputs'], indent=2)}"
                )
            if hd.get("data_attributes"):
                parts.append(
                    f"\nData Attributes:\n{json.dumps(hd['data_attributes'][:10], indent=2)}"
                )

        if context.get("files"):
            parts.append("\n=== FILES ANALYSIS ===")
            for i, file_info in enumerate(context["files"][:5]):
                parts.append(f"\n--- File {i+1}: {file_info.get('href', '')} ---")
                parts.append(f"Type: {file_info.get('kind', '')}")
                summary = file_info.get("summary", {})
                parts.append(json.dumps(summary, indent=2)[:4000])

        if context.get("audio_transcripts"):
            parts.append("\n=== AUDIO TRANSCRIPTS ===")
            for i, audio in enumerate(context["audio_transcripts"]):
                parts.append(f"\nAudio {i+1} ({audio.get('href', '')}):")
                parts.append(audio.get("transcript", ""))

        full_text = "\n".join(parts)
        if len(full_text) > 25000:
            full_text = full_text[:25000] + "\n\n[Content truncated due to length]"
        return full_text

    # ------------------------------------------------------------------
    # Fallback answer
    # ------------------------------------------------------------------
    def _fallback_answer(self, context: Dict[str, Any]) -> Any:
        page_text = context.get("page_text", "")

        patterns = [
            r"answer[:\s]+(\d+\.?\d*)",
            r"result[:\s]+(\d+\.?\d*)",
            r"total[:\s]+(\d+\.?\d*)",
            r"sum[:\s]+(\d+\.?\d*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                val = match.group(1)
                try:
                    return float(val)
                except Exception:
                    return val

        nums = re.findall(r"-?\d+\.?\d*", page_text)
        if nums:
            val = nums[-1]
            try:
                return float(val)
            except Exception:
                return val

        return "unable_to_determine"

    # ------------------------------------------------------------------
    # Solve ONE page
    # ------------------------------------------------------------------
    def _solve_one_page(
        self, page: Page, url: str, time_left: float
    ) -> Tuple[Any, Optional[str], Optional[str], Dict[str, Any]]:
        if time_left < 10:
            return None, None, None, {"error": "timeout"}

        h = hashlib.md5(url.encode()).hexdigest()
        if h in self.visited:
            return None, None, None, {"error": "already_visited"}
        self.visited.add(h)

        self._debug("=" * 60)
        self._debug(f"ANALYZING PAGE: {url}")
        self._debug("=" * 60)

        try:
            page.goto(url, wait_until="networkidle", timeout=50000)
            page.wait_for_timeout(1500)
        except Exception as e:
            self._debug("Page load error:", e)
            try:
                page.wait_for_timeout(2000)
            except Exception:
                pass

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(separator="\n", strip=True)

        hidden_data = self._extract_hidden_data(html, page)
        submit_url = self._find_submit_url(html, url)

        # Links (for context)
        links_summary = []
        for a in soup.find_all("a", href=True):
            links_summary.append(
                {"text": (a.get_text(strip=True) or "")[:300], "href": a["href"]}
            )

        file_summaries: List[Dict[str, Any]] = []
        audio_transcripts: List[Dict[str, Any]] = []
        csv_paths: List[str] = []

        # Detect & process files
        for a in soup.find_all("a", href=True):
            href = a["href"]
            lower = href.lower()
            if not any(
                lower.endswith(x)
                for x in (
                    ".csv",
                    ".xlsx",
                    ".xls",
                    ".pdf",
                    ".json",
                    ".mp3",
                    ".wav",
                    ".ogg",
                    ".m4a",
                    ".flac",
                    ".txt",
                    ".log",
                )
            ):
                continue

            dl = self._download_file(href, url)
            if not dl:
                continue
            path, content, ftype = dl

            if ftype in ("csv", "xlsx"):
                if ftype == "csv":
                    csv_paths.append(path)
                file_summaries.append(
                    {
                        "href": href,
                        "kind": ftype,
                        "path": path,
                        "summary": self._summarise_csv_xlsx(path),
                    }
                )
            elif ftype == "pdf":
                file_summaries.append(
                    {
                        "href": href,
                        "kind": "pdf",
                        "path": path,
                        "summary": self._summarise_pdf(path),
                    }
                )
            elif ftype == "json":
                file_summaries.append(
                    {
                        "href": href,
                        "kind": "json",
                        "path": path,
                        "summary": self._summarise_json_file(path),
                    }
                )
            elif ftype == "text":
                file_summaries.append(
                    {
                        "href": href,
                        "kind": "text",
                        "path": path,
                        "summary": self._summarise_text_file(path),
                    }
                )
            elif ftype == "audio":
                transcript = self._transcribe_audio_llm(
                    content, os.path.splitext(path)[1]
                )
                if transcript:
                    audio_transcripts.append(
                        {"href": href, "transcript": transcript}
                    )

        # Build context
        context: Dict[str, Any] = {
            "page_url": url,
            "page_text": page_text,
            "hidden_data": hidden_data,
            "links": links_summary,
            "files": file_summaries,
            "audio_transcripts": audio_transcripts,
        }

        # --- SPECIAL: audio + CSV rule-based solver (for cutoff problems) ---
        answer: Any = None
        if audio_transcripts and csv_paths:
            self._debug("Trying rule-based audio+CSV solver...")
            for audio in audio_transcripts:
                for csv_path in csv_paths:
                    ans = self._try_audio_csv_solve(audio["transcript"], csv_path)
                    if ans is not None:
                        answer = ans
                        break
                if answer is not None:
                    break

        # General LLM reasoning if still unanswered
        if answer is None:
            answer = self._llm_solve_from_context(context)

        if answer is None:
            self._debug("LLM failed, using fallback logic...")
            answer = self._fallback_answer(context)

        self._debug("FINAL ANSWER:", answer)

        # Submit
        submit_resp: Dict[str, Any] = {"error": "no-submit-url"}
        next_url: Optional[str] = None

        if submit_url:
            payload = {
                "email": self.email,
                "secret": self.secret,
                "url": url,
                "answer": answer,
            }
            self._debug("Submitting to:", submit_url)
            try:
                r = self.session.post(submit_url, json=payload, timeout=30)
                try:
                    submit_resp = r.json()
                except Exception:
                    submit_resp = {"text": r.text[:500]}
                if isinstance(submit_resp, dict):
                    next_url = submit_resp.get("url")
                    self._debug("Next URL:", next_url)
            except Exception as e:
                submit_resp = {"error": str(e)}

        return answer, submit_url, next_url, submit_resp

    # ------------------------------------------------------------------
    # Public orchestrator
    # ------------------------------------------------------------------
    def solve_and_submit(self, url: str, time_budget_sec: int = 180) -> Dict[str, Any]:
        start = time.time()
        result: Dict[str, Any] = {
            "first": None,
            "submissions": [],
            "completed": False,
            "total_time": 0.0,
        }

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                )
            except Exception as e:
                return {"error": f"browser-launch-failed: {e}"}

            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            page.set_default_timeout(50000)

            # First page
            elapsed = time.time() - start
            ans, s_url, nxt, resp = self._solve_one_page(
                page, url, time_budget_sec - elapsed
            )
            result["first"] = {
                "url": url,
                "submit_url": s_url,
                "answer": ans,
                "submit_response": resp,
            }

            # Follow chain
            steps = 0
            max_steps = 50
            while nxt and steps < max_steps:
                elapsed = time.time() - start
                if elapsed > time_budget_sec - 15:
                    self._debug("Time budget nearly exhausted")
                    break

                steps += 1
                self._debug("\n" + "=" * 60)
                self._debug(f"CHAIN STEP {steps}: {nxt}")
                self._debug("=" * 60 + "\n")

                ans2, s_url2, nxt2, resp2 = self._solve_one_page(
                    page, nxt, time_budget_sec - elapsed
                )

                if ans2 is None and resp2.get("error") == "already_visited":
                    self._debug("Breaking due to loop detection")
                    break

                result["submissions"].append(
                    {
                        "step": steps,
                        "url": nxt,
                        "answer": ans2,
                        "submit_url": s_url2,
                        "submit_response": resp2,
                    }
                )
                nxt = nxt2

            browser.close()
            result["completed"] = nxt is None
            result["total_time"] = time.time() - start
            result["total_steps"] = steps

        return result


# Alias for backward compatibility with app.py
QuizSolver = EnhancedQuizSolver


if __name__ == "__main__":
    EMAIL = os.getenv("QUIZ_EMAIL", "your-email@example.com")
    SECRET = os.getenv("QUIZ_SECRET", "your-secret")
    START_URL = os.getenv("QUIZ_URL", "https://quiz.example.com/start")

    solver = QuizSolver(email=EMAIL, secret=SECRET)
    result = solver.solve_and_submit(START_URL)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
