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


class QuizSolver:
    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.session = requests.Session()
        self.visited = set()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _debug(self, *args):
        print("[DEBUG]", *args, flush=True)

    def _find_submit_url(self, html: str, page_url: str) -> Optional[str]:
        """Find /submit endpoint."""
        m = re.search(r"https?://[^\s\"'<>]+/submit", html)
        if m:
            return m.group(0)
        if "/submit" in html:
            p = urlparse(page_url)
            return f"{p.scheme}://{p.netloc}/submit"
        return None

    def _decode_base64(self, s: str) -> Optional[str]:
        try:
            s = s.replace("-", "+").replace("_", "/")
            pad = len(s) % 4
            if pad:
                s += "=" * (4 - pad)
            return base64.b64decode(s).decode("utf-8", "ignore")
        except Exception:
            return None

    def _extract_atob_chunks(self, html: str) -> List[str]:
        """Decode atob(...) strings (hidden text / JSON)."""
        chunks = []
        pattern = r'atob\(\s*[`"\']([^`"\']+)[`"\']\s*\)'
        for m in re.finditer(pattern, html):
            dec = self._decode_base64(m.group(1))
            if dec:
                chunks.append(dec)
            if len(chunks) >= 10:
                break
        return chunks

    # ------------------------------------------------------------------
    # Download + file summarisation
    # ------------------------------------------------------------------
    def _download_file(self, href: str, base_url: str) -> Optional[Tuple[str, bytes, str]]:
        if not href:
            return None
        if not href.lower().startswith("http"):
            href = urljoin(base_url, href)

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
        elif "excel" in ct or lower_href.endswith(".xlsx"):
            ext = ".xlsx"
            ftype = "xlsx"
        elif "json" in ct or lower_href.endswith(".json"):
            ext = ".json"
            ftype = "json"
        elif any(lower_href.endswith(x) for x in [".mp3", ".wav", ".ogg", ".m4a", ".flac"]):
            ext = "." + lower_href.split(".")[-1].split("?")[0]
            ftype = "audio"
        else:
            name = href.split("/")[-1]
            if "." in name:
                ext = "." + name.split(".")[-1].split("?")[0]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(content)
        tmp.close()
        return tmp.name, content, ftype

    def _summarise_csv_xlsx(self, path: str) -> Dict[str, Any]:
        """Return compact JSON summary of CSV/XLSX."""
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, nrows=2000)
            else:
                df = pd.read_excel(path, nrows=2000)
        except Exception as e:
            self._debug("CSV/XLSX read error:", e)
            return {"type": "table", "error": str(e)}

        # Compact representation: columns + first 30 rows
        max_rows = 30
        preview = df.head(max_rows)
        rows = preview.to_dict(orient="records")

        summary = {
            "type": "table",
            "columns": list(df.columns),
            "n_rows_sampled": len(rows),
            "rows": rows,
        }
        return summary

    def _summarise_pdf(self, path: str) -> Dict[str, Any]:
        """Extract text & simple tables from first few pages of PDF."""
        out: Dict[str, Any] = {"type": "pdf", "pages": []}
        try:
            with pdfplumber.open(path) as pdf:
                max_pages = min(len(pdf.pages), 5)
                for i in range(max_pages):
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    text = text[:4000]  # limit per page
                    tables_data = []
                    tables = page.extract_tables()
                    if tables:
                        for t in tables[:3]:
                            tables_data.append(t[:10])  # at most 10 rows
                    out["pages"].append(
                        {"page_index": i, "text": text, "tables": tables_data}
                    )
        except Exception as e:
            self._debug("PDF summary error:", e)
            out["error"] = str(e)
        return out

    def _summarise_json_file(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._debug("JSON file read error:", e)
            return {"type": "json", "error": str(e)}

        # Truncate if very large
        try:
            text = json.dumps(data)
            if len(text) > 8000:
                # show keys/top structure, not full
                if isinstance(data, dict):
                    data = {"keys": list(data.keys())[:100]}
                elif isinstance(data, list):
                    data = data[:50]
        except Exception:
            pass

        return {"type": "json", "data": data}

    # ------------------------------------------------------------------
    # Audio transcription via AI Pipe LLM
    # ------------------------------------------------------------------
    def _transcribe_audio_llm(self, audio_bytes: bytes, ext: str) -> Optional[str]:
        if not AIPIPE_TOKEN:
            self._debug("No AIPIPE_TOKEN set; cannot transcribe audio")
            return None

        self._debug("Transcribing audio via AIPipe...")
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
                    "content": "You are a perfect speech-to-text engine. Return ONLY the exact transcript.",
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
            r = self.session.post(url, headers=headers, json=payload, timeout=60)
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                # openrouter may return list-of-parts
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                transcript = " ".join(text_parts).strip()
            else:
                transcript = str(content).strip()
            self._debug("Audio transcript:", transcript[:200])
            return transcript
        except Exception as e:
            self._debug("AIPipe audio error:", e)
            return None

    # ------------------------------------------------------------------
    # LLM reasoning over whole context
    # ------------------------------------------------------------------
    def _llm_solve_from_context(self, context: Dict[str, Any]) -> Any:
        """
        Send page+files+audio context to LLM.
        LLM must return JSON like {"answer": ...}.
        """
        if not AIPIPE_TOKEN:
            self._debug("No AIPIPE_TOKEN set; cannot use LLM")
            return None

        url = "https://aipipe.org/openrouter/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        }

        # Compact JSON string for context
        context_json = json.dumps(context, ensure_ascii=False)
        if len(context_json) > 20000:
            context_json = context_json[:20000]  # hard cap

        system_prompt = (
            "You are solving data-analysis quiz questions from IITM TDS. "
            "You will receive a JSON object describing a quiz webpage: page text, "
            "HTML snippet, links, decoded scripts, file summaries (tables, PDFs, JSON), "
            "and audio transcripts with instructions.\n\n"
            "Your job: Carefully read all instructions and data, perform any required "
            "calculations or reasoning, and determine the FINAL ANSWER.\n\n"
            "OUTPUT FORMAT (IMPORTANT): Return ONLY a valid JSON object with a single key 'answer'. "
            "For example: {\"answer\": 12345} or {\"answer\": \"some string\"} or "
            "{\"answer\": true} or {\"answer\": {\"x\": 1, \"y\": 2}}. "
            "Do NOT include explanations, extra keys, or plain text outside JSON."
        )

        payload = {
            "model": "openai/gpt-4.1-nano",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Here is the quiz page context JSON:\n{context_json}",
                },
            ],
        }

        try:
            r = self.session.post(url, headers=headers, json=payload, timeout=90)
            data = r.json()
            content = data["choices"][0]["message"]["content"]

            if isinstance(content, str):
                answer_obj = json.loads(content)
            else:
                # Sometimes it's already parsed as dict
                answer_obj = content

            self._debug("LLM raw answer_obj:", answer_obj)
            if isinstance(answer_obj, dict) and "answer" in answer_obj:
                return answer_obj["answer"]
        except Exception as e:
            self._debug("AIPipe reasoning error:", e)

        return None

    # ------------------------------------------------------------------
    # Solve ONE page (build context → LLM → submit)
    # ------------------------------------------------------------------
    def _solve_one_page(
        self, page: Page, url: str, time_left: float
    ) -> Tuple[Any, Optional[str], Optional[str], Dict[str, Any]]:
        if time_left < 10:
            return None, None, None, {"error": "timeout"}

        # protect from infinite loops
        h = hashlib.md5(url.encode()).hexdigest()
        if h in self.visited:
            return None, None, None, {"error": "visited"}
        self.visited.add(h)

        self._debug("=== Visiting ===", url)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=40000)
            page.wait_for_timeout(800)
        except Exception as e:
            self._debug("Page load error:", e)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(separator="\n", strip=True)

        submit_url = self._find_submit_url(html, url)
        atob_chunks = self._extract_atob_chunks(html)

        # Collect links
        links_summary = []
        for a in soup.find_all("a", href=True):
            links_summary.append(
                {
                    "text": (a.get_text(strip=True) or "")[:200],
                    "href": a["href"],
                }
            )
            if len(links_summary) >= 50:
                break

        # Detect & summarise files
        file_summaries = []
        audio_transcripts = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            lower = href.lower()
            if any(lower.endswith(x) for x in (".csv", ".xlsx", ".pdf", ".json", ".mp3", ".wav", ".ogg", ".m4a", ".flac")):
                dl = self._download_file(href, url)
                if not dl:
                    continue
                path, content, ftype = dl

                if ftype in ("csv", "xlsx"):
                    file_summaries.append(
                        {
                            "href": href,
                            "kind": ftype,
                            "summary": self._summarise_csv_xlsx(path),
                        }
                    )
                elif ftype == "pdf":
                    file_summaries.append(
                        {
                            "href": href,
                            "kind": "pdf",
                            "summary": self._summarise_pdf(path),
                        }
                    )
                elif ftype == "json":
                    file_summaries.append(
                        {
                            "href": href,
                            "kind": "json",
                            "summary": self._summarise_json_file(path),
                        }
                    )
                elif ftype == "audio":
                    transcript = self._transcribe_audio_llm(content, os.path.splitext(path)[1])
                    audio_transcripts.append(
                        {"href": href, "transcript": transcript}
                    )

        # Short HTML snippet
        html_snippet = html[:8000]

        # Build full context for LLM
        context: Dict[str, Any] = {
            "page_url": url,
            "page_text": page_text[:12000],
            "html_snippet": html_snippet,
            "links": links_summary,
            "decoded_scripts": atob_chunks[:10],
            "files": file_summaries,
            "audio_transcripts": audio_transcripts,
        }

        # LLM reasoning
        answer = self._llm_solve_from_context(context)

        # Fallback: last number on page, if LLM failed
        if answer is None:
            nums = re.findall(r"\d+\.?\d*", page_text)
            if nums:
                try:
                    answer = float(nums[-1])
                except Exception:
                    answer = nums[-1]
            else:
                answer = "default"

        self._debug("FINAL ANSWER:", answer)

        # Submit to quiz engine
        submit_resp: Dict[str, Any] = {"error": "no-submit-url"}
        next_url = None

        if submit_url:
            payload = {
                "email": self.email,
                "secret": self.secret,
                "url": url,
                "answer": answer,
            }
            self._debug("Submitting to:", submit_url, "payload:", payload)

            try:
                r = self.session.post(submit_url, json=payload, timeout=30)
                try:
                    submit_resp = r.json()
                except Exception:
                    submit_resp = {"text": r.text}
                if isinstance(submit_resp, Dict):
                    next_url = submit_resp.get("url")
            except Exception as e:
                submit_resp = {"error": str(e)}

        return answer, submit_url, next_url, submit_resp

    # ------------------------------------------------------------------
    # Public orchestrator
    # ------------------------------------------------------------------
    def solve_and_submit(self, url: str, time_budget_sec: int = 170) -> Dict[str, Any]:
        start = time.time()
        result: Dict[str, Any] = {
            "first": None,
            "submissions": [],
            "completed": False,
        }

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )
            except Exception as e:
                return {"error": f"browser-launch-failed: {e}"}

            page = browser.new_page()
            page.set_default_timeout(40000)

            # First URL
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

            # Chain of follow-up URLs
            steps = 0
            while nxt and steps < 40:
                elapsed = time.time() - start
                if elapsed > 165:
                    self._debug("Global time limit reached")
                    break

                steps += 1
                self._debug(f"--- Chain step {steps}: {nxt} ---")
                ans2, s_url2, nxt2, resp2 = self._solve_one_page(
                    page, nxt, time_budget_sec - elapsed
                )
                if ans2 is None:
                    break
                result["submissions"].append(
                    {
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

        return result
