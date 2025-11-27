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

# AI Pipe token – set this as an env var in Railway
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")


class QuizSolver:
    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.session = requests.Session()
        self.visited = set()

    def _debug(self, *args):
        print("[DEBUG]", *args, flush=True)

    # -----------------------------------------------------
    # HTML / URL helpers
    # -----------------------------------------------------
    def _find_submit_url(self, html: str, page_url: str) -> Optional[str]:
        """Find the /submit URL."""
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
        """Find atob("...") payloads and decode them."""
        chunks = []
        pattern = r'atob\(\s*[`"\']([^`"\']+)[`"\']\s*\)'
        for m in re.finditer(pattern, html):
            dec = self._decode_base64(m.group(1))
            if dec:
                chunks.append(dec)
            if len(chunks) >= 5:
                break
        return chunks

    # -----------------------------------------------------
    # File download + processing
    # -----------------------------------------------------
    def _download_file(self, href: str, base_url: str) -> Optional[Tuple[str, bytes, str]]:
        """Download a file and return (path, bytes, type)."""
        if not href:
            return None
        if not href.lower().startswith("http"):
            href = urljoin(base_url, href)

        self._debug("Downloading file:", href)
        try:
            r = self.session.get(href, timeout=25)
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

        if "pdf" in ct or href.lower().endswith(".pdf"):
            ext = ".pdf"
            ftype = "pdf"
        elif "csv" in ct or href.lower().endswith(".csv"):
            ext = ".csv"
            ftype = "csv"
        elif "excel" in ct or href.lower().endswith(".xlsx"):
            ext = ".xlsx"
            ftype = "xlsx"
        elif "json" in ct or href.lower().endswith(".json"):
            ext = ".json"
            ftype = "json"
        elif any(href.lower().endswith(x) for x in [".mp3", ".wav", ".ogg", ".m4a", ".flac"]):
            ext = "." + href.split(".")[-1].split("?")[0]
            ftype = "audio"
        else:
            # unknown – still save with whatever extension is there
            name = href.split("/")[-1]
            if "." in name:
                ext = "." + name.split(".")[-1].split("?")[0]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(content)
        tmp.close()
        return tmp.name, content, ftype

    def _process_pdf_sum(self, path: str) -> Optional[float]:
        """Sum 'value'-like column in PDF tables."""
        self._debug("Processing PDF:", path)
        try:
            with pdfplumber.open(path) as pdf:
                total = 0.0
                found = False
                for page in pdf.pages[:10]:
                    tables = page.extract_tables()
                    if not tables:
                        continue
                    for t in tables:
                        if not t or len(t) < 2:
                            continue
                        header = [str(h).strip().lower() if h else "" for h in t[0]]
                        idx = -1
                        for i, h in enumerate(header):
                            if h in ("value", "amount", "total", "sum"):
                                idx = i
                                break
                        if idx == -1:
                            continue
                        for row in t[1:]:
                            try:
                                total += float(row[idx])
                                found = True
                            except Exception:
                                pass
                if found:
                    return total
        except Exception as e:
            self._debug("PDF error:", e)
        return None

    def _process_csv_xlsx_sum(self, path: str) -> Optional[float]:
        """Sum value-like column in CSV/XLSX."""
        self._debug("Processing CSV/XLSX:", path)
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, nrows=20000)
            else:
                df = pd.read_excel(path, nrows=20000)

            cols = [c.lower().strip() for c in df.columns]
            for key in ("value", "amount", "total", "sum"):
                if key in cols:
                    idx = cols.index(key)
                    s = pd.to_numeric(df.iloc[:, idx], errors="coerce").sum()
                    return float(s)

            nums = df.select_dtypes(include=["number"])
            if not nums.empty:
                return float(nums.sum().sum())
        except Exception as e:
            self._debug("CSV/XLSX error:", e)
        return None

    def _process_json_answer(self, path: str) -> Optional[Any]:
        """Extract answer from JSON if possible."""
        self._debug("Processing JSON:", path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "answer" in data:
                    return data["answer"]
                if "values" in data and isinstance(data["values"], list):
                    return sum(v for v in data["values"] if isinstance(v, (int, float)))
                return sum(v for v in data.values() if isinstance(v, (int, float)))
        except Exception as e:
            self._debug("JSON error:", e)
        return None

    # -----------------------------------------------------
    # Audio via AI Pipe LLM
    # -----------------------------------------------------
    def _llm_transcribe_audio(self, audio_bytes: bytes, ext: str) -> Optional[str]:
        """Send audio bytes to AIPipe + GPT-4.1-nano to get transcript."""
        if not AIPIPE_TOKEN:
            self._debug("No AIPIPE_TOKEN, skipping online transcription")
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
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an accurate transcription engine. "
                                "Transcribe the following audio exactly. "
                                "Return ONLY the transcript text, nothing else."
                            ),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": fmt},
                        },
                    ],
                }
            ],
        }

        try:
            r = self.session.post(url, headers=headers, json=payload, timeout=40)
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            return txt.strip()
        except Exception as e:
            self._debug("AIPipe transcription error:", e)
            return None

    def _parse_direct_number_from_text(self, text: str) -> Optional[float]:
        """Pull a direct numeric answer from transcript, if stated."""
        low = text.lower()
        patterns = [
            r"(?:secret\s*code|code|cutoff|value|number)\s*(?:is|=|:)?\s*(\d+)",
            r"(?:answer|final\s*answer)\s*(?:is|=|:)?\s*(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, low)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass

        nums = re.findall(r"\d+\.?\d*", low)
        if nums:
            try:
                return float(nums[-1])
            except Exception:
                pass
        return None

    def _execute_audio_instructions(
        self, transcript: str, page: Page, url: str
    ) -> Optional[float]:
        """
        When audio describes: 'download the CSV / Excel / PDF / JSON and
        sum/average/...', follow those instructions.
        """
        if not transcript:
            return None

        t = transcript.lower()
        self._debug("Audio transcript:", transcript)

        # operation
        op = "sum"
        if any(k in t for k in ["average", "mean"]):
            op = "avg"
        elif any(k in t for k in ["max", "highest", "largest"]):
            op = "max"
        elif any(k in t for k in ["min", "lowest", "smallest"]):
            op = "min"
        elif any(k in t for k in ["count", "number of"]):
            op = "count"

        # file type
        ftype = None
        ext = None
        if "csv" in t:
            ftype, ext = "csv", ".csv"
        elif any(k in t for k in ["excel", "xlsx", "spreadsheet"]):
            ftype, ext = "xlsx", ".xlsx"
        elif "pdf" in t:
            ftype, ext = "pdf", ".pdf"
        elif "json" in t:
            ftype, ext = "json", ".json"

        if not ext:
            return None

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        target_href = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(ext):
                target_href = href
                break

        if not target_href:
            return None

        dl = self._download_file(target_href, url)
        if not dl:
            return None
        path, _bytes, dtype = dl

        # Process the file according to dtype & op
        if dtype in ("csv", "xlsx"):
            try:
                if dtype == "csv":
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
            except Exception as e:
                self._debug("Audio CSV/XLSX read error:", e)
                return None

            nums = df.select_dtypes(include=["number"])
            if nums.empty:
                return None
            col = nums.columns[0]
            series = nums[col].dropna()

            if op == "sum":
                return float(series.sum())
            if op == "avg":
                return float(series.mean())
            if op == "max":
                return float(series.max())
            if op == "min":
                return float(series.min())
            if op == "count":
                return float(series.count())

        elif dtype == "pdf":
            return self._process_pdf_sum(path)
        elif dtype == "json":
            val = self._process_json_answer(path)
            if isinstance(val, (int, float)):
                return float(val)

        return None

    # -----------------------------------------------------
    # Answer from HTML directly
    # -----------------------------------------------------
    def _extract_answer_from_page(self, page: Page, html: str) -> Optional[Any]:
        """Use JS variables / data-answer attributes / regex."""
        try:
            ans = page.evaluate(
                """
            () => {
                if (window.answer !== undefined) return window.answer;
                if (window.solution !== undefined) return window.solution;
                if (window.result !== undefined) return window.result;
                const el = document.querySelector('[data-answer], .answer, #answer');
                if (el) {
                    if (el.dataset.answer) return el.dataset.answer;
                    return el.textContent.trim();
                }
                return null;
            }
            """
            )
            if ans is not None:
                try:
                    return float(ans)
                except Exception:
                    return ans
        except Exception:
            pass

        m = re.search(r'answer["\s:=]+(\d+\.?\d*)', html, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return m.group(1)

        return None

    # -----------------------------------------------------
    # Solve ONE page
    # -----------------------------------------------------
    def _solve_one_page(
        self, page: Page, url: str, time_left: float
    ) -> Tuple[Any, Optional[str], Optional[str], Dict[str, Any]]:
        if time_left < 5:
            return None, None, None, {"error": "timeout"}

        h = hashlib.md5(url.encode()).hexdigest()
        if h in self.visited:
            return None, None, None, {"error": "visited"}
        self.visited.add(h)

        self._debug("=== Visiting ===", url)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(600)
        except Exception as e:
            self._debug("Page load error:", e)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        submit_url = self._find_submit_url(html, url)
        atob_chunks = self._extract_atob_chunks(html)

        # Detect audio + one main data file
        audio_bytes = None
        audio_ext = ".mp3"
        data_file_path = None
        data_file_type = None

        for a in soup.find_all("a", href=True):
            href = a["href"]
            lower = href.lower()

            if (audio_bytes is None) and any(
                lower.endswith(x) for x in (".mp3", ".wav", ".ogg", ".m4a", ".flac")
            ):
                dl = self._download_file(href, url)
                if dl:
                    path, content, ftype = dl
                    if ftype == "audio":
                        audio_bytes = content
                        audio_ext = os.path.splitext(path)[1] or ".mp3"

            if (data_file_path is None) and any(
                lower.endswith(x) for x in (".pdf", ".csv", ".xlsx", ".json")
            ):
                dl = self._download_file(href, url)
                if dl:
                    data_file_path, _bytes, data_file_type = dl

        answer: Any = None

        # 1) Audio-based instructions, if any
        if audio_bytes is not None:
            transcript = self._llm_transcribe_audio(audio_bytes, audio_ext)
            if transcript:
                num = self._parse_direct_number_from_text(transcript)
                if num is not None:
                    answer = num
                else:
                    answer = self._execute_audio_instructions(transcript, page, url)

        # 2) Direct from page
        if answer is None:
            answer = self._extract_answer_from_page(page, html)

        # 3) atob JSON with answer
        if answer is None:
            for chunk in atob_chunks:
                try:
                    j = json.loads(chunk)
                    if isinstance(j, dict) and "answer" in j:
                        answer = j["answer"]
                        break
                except Exception:
                    pass

        # 4) Data file numeric
        if answer is None and data_file_path and data_file_type:
            if data_file_type == "pdf":
                answer = self._process_pdf_sum(data_file_path)
            elif data_file_type in ("csv", "xlsx"):
                answer = self._process_csv_xlsx_sum(data_file_path)
            elif data_file_type == "json":
                answer = self._process_json_answer(data_file_path)

        # 5) fallback – last number in visible text
        if answer is None:
            nums = re.findall(r"\d+\.?\d*", text)
            if nums:
                try:
                    answer = float(nums[-1])
                except Exception:
                    answer = nums[-1]

        if answer is None:
            answer = "default"

        self._debug("ANSWER:", answer)

        # Submit the answer
        submit_response: Dict[str, Any] = {"error": "no-submit-url"}
        next_url: Optional[str] = None

        if submit_url:
            payload = {
                "email": self.email,
                "secret": self.secret,
                "url": url,
                "answer": answer,
            }
            self._debug("Submitting to:", submit_url, "payload:", payload)
            try:
                r = self.session.post(submit_url, json=payload, timeout=25)
                try:
                    submit_response = r.json()
                except Exception:
                    submit_response = {"text": r.text}
                if isinstance(submit_response, dict):
                    next_url = submit_response.get("url")
            except Exception as e:
                submit_response = {"error": str(e)}

        return answer, submit_url, next_url, submit_response

    # -----------------------------------------------------
    # Public orchestrator
    # -----------------------------------------------------
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
            page.set_default_timeout(30000)

            # First quiz URL
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

            # Follow chain of URLs
            iterations = 0
            while nxt and iterations < 40:
                elapsed = time.time() - start
                if elapsed > 165:
                    self._debug("Global time limit reached")
                    break

                iterations += 1
                self._debug(f"--- Chain step {iterations}: {nxt} ---")
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
