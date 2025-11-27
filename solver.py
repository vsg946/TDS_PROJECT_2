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

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# AIPipe token/model from environment
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
# Highest reasonable model; you can override with AIPIPE_MODEL env var
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "openai/gpt-4.1")


class EnhancedQuizSolver:
    """
    Main solver class: renders quiz pages, extracts all context
    (HTML, hidden data, files, audio instructions), then either:
      - solves directly (e.g., CSV + cutoff from page / audio), or
      - asks an LLM via AIPipe to compute the final answer.

    Externally, use QuizSolver below (alias to this class).
    """

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
        """
        Robustly find the /submit endpoint without accidentally
        grabbing broken HTML fragments.
        """
        soup = BeautifulSoup(html, "html.parser")

        # 1) Look for forms whose action contains "submit"
        for form in soup.find_all("form"):
            action = (form.get("action") or "").strip()
            if not action:
                continue
            if "submit" in action.lower():
                return urljoin(page_url, action)

        # 2) Look for <a href=".../submit">
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if "submit" in href.lower():
                return urljoin(page_url, href)

        # 3) Regex for full URLs that look like /submit
        url_pattern = r"https?://[^\s\"'<>]+"
        for full in re.findall(url_pattern, html):
            if "/submit" in full:
                # Strip trailing punctuation, if any
                full_clean = full.rstrip(");,'\"")
                return full_clean

        # 4) Last resort: same-origin /submit
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
        """
        Extract hidden / implicit data from the page:
        - atob-decoded strings
        - JSON objects in scripts
        - hidden inputs
        - data-* attributes
        - JS variable assignments
        - meta tags
        - localStorage
        - cookies
        """
        hidden_data: Dict[str, Any] = {
            "atob_decoded": [],
            "json_objects": [],
            "hidden_inputs": [],
            "data_attributes": [],
            "script_variables": [],
            "meta_tags": [],
            "localStorage": {},
            "cookies": {},
        }

        soup = BeautifulSoup(html, "html.parser")

        # 1) atob() calls
        atob_pattern = r'atob\(\s*[`"\']([^`"\']+)[`"\']\s*\)'
        for m in re.finditer(atob_pattern, html):
            decoded = self._decode_base64(m.group(1))
            if decoded:
                hidden_data["atob_decoded"].append(decoded)

        # 2) JSON & variables in <script>
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

            # var/let/const assignments
            var_pattern = r"(?:var|let|const)\s+(\w+)\s*=\s*([^;]+);"
            for match in re.finditer(var_pattern, script_text):
                hidden_data["script_variables"].append(
                    {"name": match.group(1), "value": match.group(2).strip()}
                )

        # 3) Hidden inputs
        for inp in soup.find_all("input", type="hidden"):
            hidden_data["hidden_inputs"].append(
                {
                    "name": inp.get("name", ""),
                    "value": inp.get("value", ""),
                }
            )

        # 4) data-* attrs
        for elem in soup.find_all(attrs=lambda attrs: attrs and any(
            k.startswith("data-") for k in attrs
        )):
            data_attrs = {k: v for k, v in elem.attrs.items() if k.startswith("data-")}
            if data_attrs:
                hidden_data["data_attributes"].append(
                    {"tag": elem.name, "attributes": data_attrs}
                )

        # 5) meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            if name:
                hidden_data["meta_tags"].append(
                    {"name": name, "content": meta.get("content", "")}
                )

        # 6) localStorage
        try:
            storage = page.evaluate("() => JSON.stringify(localStorage)")
            hidden_data["localStorage"] = json.loads(storage)
        except Exception:
            pass

        # 7) cookies
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
            return self.cache[cache_key]

        self._debug("Downloading file:", href)
        try:
            r = self.session.get(href, timeout=30)
            if r.status_code != 200:
                self._debug("Download status:", r.status_code)
                return None
            content = r.content
        except Exception as e:
            self._debug("Download error:", e)
            return None

        ct = (r.headers.get("content-type") or "").lower()
        ext = ""
        ftype = "other"
        lower_href = href.lower()

        if "pdf" in ct or lower_href.endswith(".pdf"):
            ext, ftype = ".pdf", "pdf"
        elif "csv" in ct or lower_href.endswith(".csv"):
            ext, ftype = ".csv", "csv"
        elif "excel" in ct or any(lower_href.endswith(x) for x in [".xlsx", ".xls"]):
            ext, ftype = ".xlsx", "xlsx"
        elif "json" in ct or lower_href.endswith(".json"):
            ext, ftype = ".json", "json"
        elif any(lower_href.endswith(x) for x in [".mp3", ".wav", ".ogg", ".m4a", ".flac"]):
            ext = "." + lower_href.split(".")[-1].split("?")[0]
            ftype = "audio"
        elif any(lower_href.endswith(x) for x in [".txt", ".log"]):
            ext, ftype = ".txt", "text"
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
        """Summarise CSV/XLSX with statistics + sample rows."""
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

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            stats: Dict[str, Dict[str, Any]] = {}
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
                    col_ser = pd.to_numeric(col_ser, errors="coerce")
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

        preview = df.head(50)
        summary["sample_rows"] = preview.to_dict(orient="records")

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        value_counts: Dict[str, Dict[str, int]] = {}
        for col in categorical_cols:
            if df[col].nunique() <= 20:
                value_counts[col] = df[col].value_counts().to_dict()
        if value_counts:
            summary["value_counts"] = value_counts

        return summary

    def _summarise_pdf(self, path: str) -> Dict[str, Any]:
        """Extract text + tables + numbers from first few PDF pages."""
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
    # Audio transcription + simple instruction parsing
    # ------------------------------------------------------------------
    def _transcribe_audio_llm(self, audio_bytes: bytes, ext: str) -> Optional[str]:
        if not AIPIPE_TOKEN:
            self._debug("No AIPIPE_TOKEN set; cannot transcribe audio.")
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
            "model": AIPIPE_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a perfect speech-to-text engine. "
                        "Transcribe the audio exactly, especially numbers and instructions. "
                        "Return only the transcript as text."
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
            r = self.session.post(url, headers=headers, json=payload, timeout=120)
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

    def _parse_audio_instructions(self, transcript: str) -> Dict[str, Any]:
        """
        Very lightweight parser for the most common audio instruction pattern:
        e.g. "Add all the values in column 1 of the CSV where value is
        greater than the cutoff 26779", etc.
        """
        result: Dict[str, Any] = {
            "operation": None,     # "sum", "count", ...
            "column_index": None,  # zero-based
            "column_name": None,   # optional
            "cutoff": None,        # numeric cutoff if mentioned
            "comparison": None,    # ">", ">=", "<", "<="
        }

        t = transcript.lower()

        # Operation (very simple)
        if any(w in t for w in ["sum", "add up", "add all", "total"]):
            result["operation"] = "sum"
        elif "count" in t:
            result["operation"] = "count"

        # Column index
        m = re.search(r"column\s+(one|two|three|four|five|six|seven|eight|nine|\d+)", t)
        if m:
            token = m.group(1)
            mapping = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9,
            }
            if token.isdigit():
                idx = int(token)
            else:
                idx = mapping.get(token, 1)
            result["column_index"] = max(idx - 1, 0)

        # Column name (e.g. "value column", "values column")
        m2 = re.search(r"(?:column|field)\s+named\s+([a-zA-Z0-9_ ]+)", t)
        if m2:
            result["column_name"] = m2.group(1).strip()

        # Cutoff and comparison
        # Look explicitly for "cutoff"
        m3 = re.search(r"cutoff[^0-9\-]*(-?\d+)", t)
        if m3:
            try:
                result["cutoff"] = float(m3.group(1))
                result["comparison"] = ">"
            except Exception:
                pass
        else:
            # generic "greater than 123"
            m4 = re.search(r"greater than[^0-9\-]*(-?\d+)", t)
            if m4:
                try:
                    result["cutoff"] = float(m4.group(1))
                    result["comparison"] = ">"
                except Exception:
                    pass

        return result

    def _compute_from_csv_instructions(
        self,
        transcript: str,
        csv_path: str,
        page_text: str,
        csv_summary: Dict[str, Any],
    ) -> Optional[float]:
        """
        Attempt to solve the 'demo-audio' CSV problem WITHOUT LLM by
        using the transcript + CSV file.

        Typical pattern:
        - page text includes "Cutoff: 26779"
        - audio says 'add all values in first column greater than the cutoff'
        """
        self._debug("Trying direct CSV solve from audio instructions...")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self._debug("Direct CSV read failed:", e)
            return None

        instr = self._parse_audio_instructions(transcript)

        # Column selection
        col = None
        cols = list(df.columns)

        # Use column name if found
        if instr.get("column_name"):
            # best fuzzy-ish match
            cname = instr["column_name"].lower().replace(" ", "")
            for c in cols:
                if cname in c.lower().replace(" ", ""):
                    col = c
                    break

        # Or use column index if specified
        if col is None and instr.get("column_index") is not None:
            idx = instr["column_index"]
            if 0 <= idx < len(cols):
                col = cols[idx]

        # Fallback: first numeric column
        if col is None:
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if num_cols:
                col = num_cols[0]

        if col is None:
            self._debug("No suitable numeric column found for direct solve.")
            return None

        # Determine cutoff:
        cutoff = instr.get("cutoff")

        if cutoff is None:
            # Try to pick from page text: "Cutoff: 26779"
            m = re.search(r"cutoff[^0-9\-]*(-?\d+)", page_text.lower())
            if m:
                try:
                    cutoff = float(m.group(1))
                except Exception:
                    cutoff = None

        # No cutoff => we might just sum the whole column
        col_ser = pd.to_numeric(df[col], errors="coerce").dropna()
        if col_ser.empty:
            self._debug("Target column is empty or non-numeric.")
            return None

        if instr.get("operation") == "sum":
            if cutoff is not None:
                comp = instr.get("comparison") or ">"
                if comp == ">":
                    subset = col_ser[col_ser > cutoff]
                elif comp == ">=":
                    subset = col_ser[col_ser >= cutoff]
                elif comp == "<":
                    subset = col_ser[col_ser < cutoff]
                elif comp == "<=":
                    subset = col_ser[col_ser <= cutoff]
                else:
                    subset = col_ser
                s = float(subset.sum())
            else:
                s = float(col_ser.sum())
            self._debug("Direct CSV sum computed:", s)
            return s

        if instr.get("operation") == "count":
            if cutoff is not None:
                comp = instr.get("comparison") or ">"
                if comp == ">":
                    count = int((col_ser > cutoff).sum())
                elif comp == ">=":
                    count = int((col_ser >= cutoff).sum())
                elif comp == "<":
                    count = int((col_ser < cutoff).sum())
                elif comp == "<=":
                    count = int((col_ser <= cutoff).sum())
                else:
                    count = int(col_ser.count())
            else:
                count = int(col_ser.count())
            self._debug("Direct CSV count computed:", count)
            return float(count)

        # If operation wasn't understood, we bail out
        return None

    def _try_audio_csv_solve(
        self,
        context: Dict[str, Any],
    ) -> Optional[float]:
        """
        Look for an audio transcript + CSV in the same page and try a
        fully deterministic numeric solution before calling LLM.
        """
        if not context.get("audio_transcripts"):
            return None
        if not context.get("files"):
            return None

        # Take first audio transcript
        audio_info = context["audio_transcripts"][0]
        transcript = audio_info.get("transcript") or ""
        if not transcript:
            return None

        # Find CSV file (or XLSX) in files
        csv_file = None
        csv_summary = None
        for f in context["files"]:
            if f.get("kind") in ("csv", "xlsx"):
                csv_file = f.get("path")  # we will store path below
                csv_summary = f.get("summary")
                break

        if not csv_file:
            return None

        # Use page_text for cutoff detection
        page_text = context.get("page_text", "")

        try:
            return self._compute_from_csv_instructions(
                transcript=transcript,
                csv_path=csv_file,
                page_text=page_text,
                csv_summary=csv_summary or {},
            )
        except Exception as e:
            self._debug("Error in audio CSV solve:", e)
            return None

    # ------------------------------------------------------------------
    # LLM reasoning
    # ------------------------------------------------------------------
    def _llm_solve_from_context(self, context: Dict[str, Any]) -> Any:
        """
        Ask AIPipe / LLM to compute the answer from full context.
        """
        if not AIPIPE_TOKEN:
            self._debug("No AIPIPE_TOKEN set; cannot use LLM.")
            return None

        url = "https://aipipe.org/openrouter/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        }

        context_text = self._build_context_text(context)

        system_prompt = (
            "You are an expert data analyst solving quiz questions. "
            "You will receive:\n"
            "1. Page content (text, HTML-derived)\n"
            "2. Hidden data (decoded scripts, JSON, hidden inputs, data attributes)\n"
            "3. File summaries (CSV statistics, PDF content, JSON structure)\n"
            "4. Audio transcripts with instructions\n\n"
            "Your task:\n"
            "- Understand the question being asked\n"
            "- Locate relevant data (cutoff values, column names, conditions)\n"
            "- Perform the required calculations (sums, counts, filters, etc.)\n"
            "- Return ONLY the final answer in JSON.\n\n"
            "CRITICAL: respond with VALID JSON only, exactly in this format:\n"
            '{"answer": <your_answer>}\n'
            "Examples:\n"
            '- {"answer": 42}\n'
            '- {"answer": "YELLOW"}\n'
            '- {"answer": 3.14159}\n'
            '- {"answer": true}\n'
            "Do not include explanations or extra fields."
        )

        payload = {
            "model": AIPIPE_MODEL,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Quiz context:\n\n{context_text}",
                },
            ],
        }

        try:
            r = self.session.post(url, headers=headers, json=payload, timeout=150)
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
        """Create a structured text for the LLM from the gathered context."""
        parts: List[str] = []

        parts.append("=== PAGE INFORMATION ===")
        parts.append(f"URL: {context.get('page_url', '')}")
        parts.append("\nVISIBLE TEXT:\n")
        parts.append(context.get("page_text", "")[:8000])

        hd = context.get("hidden_data") or {}
        if hd:
            parts.append("\n=== HIDDEN DATA ===")

            if hd.get("atob_decoded"):
                parts.append("\nBase64-Decoded Snippets:\n")
                parts.append(json.dumps(hd["atob_decoded"], indent=2)[:2000])

            if hd.get("script_variables"):
                parts.append("\nJavaScript Variables (sample):\n")
                parts.append(json.dumps(hd["script_variables"][:20], indent=2)[:3000])

            if hd.get("hidden_inputs"):
                parts.append("\nHidden Form Inputs:\n")
                parts.append(json.dumps(hd["hidden_inputs"], indent=2)[:2000])

            if hd.get("data_attributes"):
                parts.append("\nData Attributes (sample):\n")
                parts.append(json.dumps(hd["data_attributes"][:10], indent=2)[:2000])

        files = context.get("files") or []
        if files:
            parts.append("\n=== FILES ANALYSIS ===")
            for i, f in enumerate(files[:5]):
                parts.append(f"\n--- File {i+1}: {f.get('href', '')} ---")
                parts.append(f"Type: {f.get('kind', '')}")
                summary = f.get("summary") or {}
                parts.append(json.dumps(summary, indent=2)[:4000])

        audios = context.get("audio_transcripts") or []
        if audios:
            parts.append("\n=== AUDIO TRANSCRIPTS ===")
            for i, a in enumerate(audios):
                parts.append(f"\nAudio {i+1} ({a.get('href', '')}):\n")
                parts.append((a.get("transcript") or "")[:4000])

        text = "\n".join(parts)
        if len(text) > 25000:
            text = text[:25000] + "\n\n[Content truncated due to length]"
        return text

    # ------------------------------------------------------------------
    # Fallback answer (no LLM / no deterministic path)
    # ------------------------------------------------------------------
    def _fallback_answer(self, context: Dict[str, Any]) -> Any:
        page_text = context.get("page_text", "")

        # Look for labelled numeric answers
        patterns = [
            r"answer[:\s]+(\d+\.?\d*)",
            r"result[:\s]+(\d+\.?\d*)",
            r"total[:\s]+(\d+\.?\d*)",
            r"sum[:\s]+(\d+\.?\d*)",
        ]
        for pattern in patterns:
            m = re.search(pattern, page_text, re.IGNORECASE)
            if m:
                val = m.group(1)
                try:
                    return float(val)
                except Exception:
                    return val

        # Last resort: last number on page
        nums = re.findall(r"-?\d+\.?\d*", page_text)
        if nums:
            last = nums[-1]
            try:
                return float(last)
            except Exception:
                return last

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
        self._debug("ANALYZING PAGE:", url)
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

        # Collect link list (for context only)
        links_summary = [
            {"text": (a.get_text(strip=True) or "")[:200], "href": a["href"]}
            for a in soup.find_all("a", href=True)
        ]

        # Files & audio
        file_summaries: List[Dict[str, Any]] = []
        audio_transcripts: List[Dict[str, Any]] = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            lower = href.lower()

            if not any(
                lower.endswith(ext)
                for ext in (
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
                summary = self._summarise_csv_xlsx(path)
                file_summaries.append(
                    {"href": href, "kind": ftype, "summary": summary, "path": path}
                )
            elif ftype == "pdf":
                summary = self._summarise_pdf(path)
                file_summaries.append(
                    {"href": href, "kind": "pdf", "summary": summary, "path": path}
                )
            elif ftype == "json":
                summary = self._summarise_json_file(path)
                file_summaries.append(
                    {"href": href, "kind": "json", "summary": summary, "path": path}
                )
            elif ftype == "text":
                summary = self._summarise_text_file(path)
                file_summaries.append(
                    {"href": href, "kind": "text", "summary": summary, "path": path}
                )
            elif ftype == "audio":
                ext = os.path.splitext(path)[1]
                transcript = self._transcribe_audio_llm(content, ext)
                if transcript:
                    audio_transcripts.append(
                        {"href": href, "transcript": transcript, "path": path}
                    )

        # Build context used by both deterministic logic & LLM
        context: Dict[str, Any] = {
            "page_url": url,
            "page_text": page_text,
            "hidden_data": hidden_data,
            "links": links_summary,
            "files": file_summaries,
            "audio_transcripts": audio_transcripts,
        }

        # 1) Try deterministic audio+CSV solve first (for demo-audio task)
        answer: Any = None
        numeric_answer = self._try_audio_csv_solve(context)
        if numeric_answer is not None:
            self._debug("Direct CSV+audio answer:", numeric_answer)
            answer = numeric_answer

        # 2) If not solved, call LLM with full context
        if answer is None:
            answer = self._llm_solve_from_context(context)

        # 3) Fallback heuristics
        if answer is None:
            self._debug("LLM failed, using fallback heuristics...")
            answer = self._fallback_answer(context)

        self._debug("FINAL ANSWER FOR PAGE:", answer)

        # Submit to quiz backend
        submit_resp: Dict[str, Any] = {"error": "no-submit-url"}
        next_url = None

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

            # Follow the chain
            steps = 0
            max_steps = 50

            while nxt and steps < max_steps:
                elapsed = time.time() - start
                if elapsed > time_budget_sec - 15:
                    self._debug("Time budget nearly exhausted, stop chaining.")
                    break

                steps += 1
                self._debug("\n" + "=" * 60)
                self._debug(f"CHAIN STEP {steps}: {nxt}")
                self._debug("=" * 60 + "\n")

                ans2, s_url2, nxt2, resp2 = self._solve_one_page(
                    page, nxt, time_budget_sec - elapsed
                )

                if ans2 is None and resp2.get("error") == "already_visited":
                    self._debug("Loop detected, breaking.")
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


# Alias expected by app.py: from solver import QuizSolver
QuizSolver = EnhancedQuizSolver


if __name__ == "__main__":
    # Local manual test (not used by Railway)
    EMAIL = os.getenv("QUIZ_EMAIL", "your-email@example.com")
    SECRET = os.getenv("QUIZ_SECRET", "your-secret")
    START_URL = os.getenv("QUIZ_URL", "https://tds-llm-analysis.s-anand.net/demo")

    solver = QuizSolver(email=EMAIL, secret=SECRET)
    res = solver.solve_and_submit(START_URL)
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(json.dumps(res, indent=2))
