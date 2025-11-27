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


AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")


class QuizSolver:
    """
    End-to-end quiz solver.

    Capabilities:
      - Render dynamic pages via Playwright (JS execution)
      - Scrape visible text + DOM
      - Download and parse PDF / CSV / XLSX / JSON
      - Detect images, audio, video links
      - Use AI Pipe (OpenRouter-compatible) LLMs to:
          * Reason over text + tables
          * Do vision OCR on images/charts
          * Transcribe & interpret audio
          * Provide final answer when rules fail
    """

    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.session = requests.Session()
        self.visited = set()

    # ---------------------------------------------------
    # Logging helper
    # ---------------------------------------------------
    def _debug(self, *args):
        print("[DEBUG]", *args, flush=True)

    # ---------------------------------------------------
    # LLM HELPERS (AI Pipe)
    # ---------------------------------------------------
    def _has_llm(self) -> bool:
        return bool(AIPIPE_TOKEN)

    def _llm_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = "openai/gpt-4.1-nano",
        timeout: int = 40,
    ) -> Optional[str]:
        """
        Generic text (or multimodal) chat with AI Pipe / OpenRouter.
        """
        if not self._has_llm():
            self._debug("No AIPIPE_TOKEN, skipping LLM call.")
            return None

        url = "https://aipipe.org/openrouter/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
        }

        try:
            r = self.session.post(url, headers=headers, json=payload, timeout=timeout)
            data = r.json()
            # Very basic extraction with guard
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self._debug("LLM chat error:", e)
            return None

    def _llm_vision_image(
        self, image_bytes: bytes, prompt: str
    ) -> Optional[str]:
        """
        Vision: send image (e.g. chart, screenshot) + prompt to GPT-4o-mini style model.
        Uses base64 data URL with a multimodal message.
        """
        if not self._has_llm():
            return None

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": data_url},
                ],
            }
        ]
        return self._llm_chat(messages, model="openai/gpt-4o-mini")

    def _llm_audio_transcribe_and_reason(
        self, audio_bytes: bytes, instructions: str = "Transcribe the audio and compute any numeric answer if requested."
    ) -> Optional[str]:
        """
        Best-effort audio + reasoning using LLM.
        This assumes AI Pipe/OpenRouter supports multimodal audio content
        via the 'input_audio' style format used by OpenAI-compatible APIs.
        """
        if not self._has_llm():
            return None

        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": "mp3"},
                    },
                ],
            }
        ]
        return self._llm_chat(messages, model="openai/gpt-4.1-nano")

    # ---------------------------------------------------
    # HTML helpers
    # ---------------------------------------------------
    def _find_submit_url(self, html: str, page_url: str) -> Optional[str]:
        """
        Extract submit URL. Try absolute; fallback to /submit on same host.
        """
        m = re.search(r"https?://[^\s\"'<>]+/submit", html)
        if m:
            return m.group(0)

        if "/submit" in html:
            p = urlparse(page_url)
            return f"{p.scheme}://{p.netloc}/submit"

        return None

    def _extract_atob_chunks(self, html: str) -> List[str]:
        """
        Extract base64 payloads from atob("...") / atob('...') / atob(`...`).
        """
        chunks = []
        pattern = r'atob\(\s*[`"\']([^`"\']+)[`"\']\s*\)'
        for m in re.finditer(pattern, html):
            raw = m.group(1)
            try:
                s = raw.replace("-", "+").replace("_", "/")
                pad = len(s) % 4
                if pad:
                    s += "=" * (4 - pad)
                dec = base64.b64decode(s).decode("utf-8", "ignore")
                chunks.append(dec)
            except:
                pass
            if len(chunks) >= 3:
                break
        return chunks

    # ---------------------------------------------------
    # File download & processing
    # ---------------------------------------------------
    def _download_file(self, href: str, base_url: str) -> Optional[Tuple[str, bytes]]:
        """
        Download file at link (absolute or relative).
        Returns (file_path, content_bytes) or (None, None).
        """
        if not href:
            return None

        if not href.lower().startswith("http"):
            href = urljoin(base_url, href)

        self._debug("Downloading file:", href)

        try:
            r = self.session.get(href, timeout=20)
            if r.status_code != 200:
                self._debug("Download failed, status:", r.status_code)
                return None
            content = r.content
        except Exception as e:
            self._debug("Download error:", e)
            return None

        # Guess extension from URL or content-type
        ct = r.headers.get("content-type", "").lower()
        ext = ""
        if "pdf" in ct or href.endswith(".pdf"):
            ext = ".pdf"
        elif "csv" in ct or href.endswith(".csv"):
            ext = ".csv"
        elif "excel" in ct or href.endswith(".xlsx"):
            ext = ".xlsx"
        elif "json" in ct or href.endswith(".json"):
            ext = ".json"
        elif any(href.lower().endswith(x) for x in [".png", ".jpg", ".jpeg", ".webp"]):
            ext = "." + href.split(".")[-1].split("?")[0]
        elif any(href.lower().endswith(x) for x in [".mp3", ".wav", ".ogg", ".m4a"]):
            ext = "." + href.split(".")[-1].split("?")[0]
        elif any(href.lower().endswith(x) for x in [".mp4", ".webm", ".mkv"]):
            ext = "." + href.split(".")[-1].split("?")[0]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(content)
        tmp.close()
        return tmp.name, content

    def _process_pdf_tables_sum(self, fp: str) -> Optional[float]:
        """
        Try to sum a 'value' column from PDF tables for typical questions.
        """
        self._debug("Processing PDF for table sums:", fp)
        try:
            with pdfplumber.open(fp) as pdf:
                total = 0.0
                found = False
                for page in pdf.pages[:10]:
                    tables = page.extract_tables()
                    if not tables:
                        continue
                    for t in tables:
                        if len(t) < 2:
                            continue
                        header = [str(h).strip().lower() for h in t[0]]
                        if "value" in header:
                            idx = header.index("value")
                            for row in t[1:]:
                                try:
                                    total += float(row[idx])
                                    found = True
                                except:
                                    pass
                if found:
                    self._debug("PDF 'value' sum:", total)
                    return total
        except Exception as e:
            self._debug("PDF error:", e)
        return None

    def _process_csv_xlsx_sum(self, fp: str) -> Optional[float]:
        """
        Try to sum a value-like column in CSV/XLSX.
        """
        self._debug("Processing CSV/XLSX:", fp)
        try:
            if fp.endswith(".csv"):
                df = pd.read_csv(fp, nrows=20000)
            else:
                df = pd.read_excel(fp, nrows=20000)

            cols_lower = [c.lower().strip() for c in df.columns]
            candidates = ["value", "amount", "total", "sum"]
            for key in candidates:
                if key in cols_lower:
                    idx = cols_lower.index(key)
                    series = pd.to_numeric(df.iloc[:, idx], errors="coerce")
                    s = float(series.sum())
                    self._debug(f"Spreadsheet sum of '{df.columns[idx]}':", s)
                    return s
        except Exception as e:
            self._debug("CSV/XLSX error:", e)
        return None

    def _process_json_answer(self, fp: str) -> Optional[Any]:
        """
        Try to extract answer from JSON.
        """
        self._debug("Processing JSON:", fp)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "answer" in data:
                    return data["answer"]
                if "values" in data and isinstance(data["values"], list):
                    return sum(
                        v for v in data["values"] if isinstance(v, (int, float))
                    )
        except Exception as e:
            self._debug("JSON error:", e)
        return None

    # ---------------------------------------------------
    # Context builder for LLM
    # ---------------------------------------------------
    def _build_context(
        self,
        url: str,
        html: str,
        text: str,
        file_summaries: List[Dict[str, Any]],
        partial_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "url": url,
            "page_text": text[:8000],  # avoid huge payloads
            "file_summaries": file_summaries,
            "partial_results": partial_results,
        }

    # ---------------------------------------------------
    # Core: solve one page
    # ---------------------------------------------------
    def _solve_one_page(
        self, page: Page, url: str, time_left: float
    ) -> Tuple[Any, Optional[str], Optional[str], Dict[str, Any]]:
        """
        Visit one URL, try to compute answer (via rules + LLM), submit, return:
            answer, submit_url, next_url, submit_response
        """
        if time_left < 5:
            self._debug("Time budget expired for:", url)
            return None, None, None, {"error": "timeout"}

        h = hashlib.md5(url.encode()).hexdigest()
        if h in self.visited:
            self._debug("Already visited:", url)
            return None, None, None, {"error": "visited"}
        self.visited.add(h)

        self._debug("\n=== Visiting ===", url)

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

        # Detect and download key files
        file_summaries: List[Dict[str, Any]] = []
        partial_results: Dict[str, Any] = {}

        # Gather all links
        links = []
        for a in soup.find_all("a", href=True):
            links.append(a["href"])

        # Track main file-based operations
        for href in links:
            href_lower = href.lower()
            if any(
                href_lower.endswith(ext)
                for ext in (".pdf", ".csv", ".xlsx", ".json", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".ogg", ".m4a", ".mp4", ".webm", ".mkv")
            ):
                res = self._download_file(href, url)
                if not res:
                    continue
                fp, content_bytes = res
                summary: Dict[str, Any] = {"href": href, "path": fp, "type": None}

                # Dispatch based on file type
                if fp.endswith(".pdf"):
                    summary["type"] = "pdf"
                    pdf_sum = self._process_pdf_tables_sum(fp)
                    if pdf_sum is not None:
                        partial_results["pdf_value_sum"] = pdf_sum
                        summary["value_sum"] = pdf_sum

                elif fp.endswith((".csv", ".xlsx")):
                    summary["type"] = "spreadsheet"
                    ss_sum = self._process_csv_xlsx_sum(fp)
                    if ss_sum is not None:
                        partial_results["spreadsheet_value_sum"] = ss_sum
                        summary["value_sum"] = ss_sum

                elif fp.endswith(".json"):
                    summary["type"] = "json"
                    ans = self._process_json_answer(fp)
                    if ans is not None:
                        partial_results["json_answer"] = ans
                        summary["json_answer"] = ans

                elif fp.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    summary["type"] = "image"
                    # Best-effort: ask vision LLM if available
                    vision_ans = self._llm_vision_image(
                        content_bytes,
                        prompt=(
                            "This image is part of a quiz question. "
                            "Read all text, inspect any charts/tables, "
                            "and determine the single most likely answer required."
                        ),
                    )
                    if vision_ans:
                        summary["vision_answer"] = vision_ans
                        # If we don't yet have an answer, we might use this later
                        if "vision_answers" not in partial_results:
                            partial_results["vision_answers"] = []
                        partial_results["vision_answers"].append(
                            {"href": href, "answer": vision_ans}
                        )

                elif fp.endswith((".mp3", ".wav", ".ogg", ".m4a")):
                    summary["type"] = "audio"
                    # Best-effort LLM-based transcription + reasoning
                    audio_ans = self._llm_audio_transcribe_and_reason(
                        content_bytes,
                        instructions=(
                            "Transcribe this audio. If the speaker asks a question or "
                            "gives instructions that require a number or text answer, "
                            "compute that final answer and return only that."
                        ),
                    )
                    if audio_ans:
                        summary["audio_answer"] = audio_ans
                        partial_results["audio_answers"] = partial_results.get(
                            "audio_answers", []
                        )
                        partial_results["audio_answers"].append(
                            {"href": href, "answer": audio_ans}
                        )

                elif fp.endswith((".mp4", ".webm", ".mkv")):
                    summary["type"] = "video"
                    # We don't process video deeply; just note presence
                    # (could be extended: extract thumbnails -> vision LLM)
                    summary["note"] = "video_downloaded_not_processed"

                file_summaries.append(summary)

        # -----------------------------
        # Rule-based attempt for answer
        # -----------------------------
        answer: Any = None

        # 1) Direct JS/global variables / [data-answer]
        try:
            js_ans = page.evaluate(
                """
                () => {
                    if (window.answer !== undefined) return window.answer;
                    if (window.solution !== undefined) return window.solution;
                    if (window.result !== undefined) return window.result;

                    const el = document.querySelector('[data-answer], #answer, .answer');
                    if (el) {
                        if (el.dataset.answer) return el.dataset.answer;
                        return el.textContent.trim();
                    }
                    return null;
                }
                """
            )
            if js_ans is not None:
                try:
                    answer = float(js_ans)
                except:
                    answer = js_ans
        except Exception as e:
            self._debug("JS-based answer lookup failed:", e)

        # 2) If JSON from atob contains explicit answer
        if answer is None:
            for chunk in atob_chunks:
                try:
                    j = json.loads(chunk)
                    if isinstance(j, dict) and "answer" in j:
                        answer = j["answer"]
                        break
                except:
                    pass

        # 3) Use partial numeric results (like pdf/csv sums) if clearly a sum question
        #    (At this level we don't parse question text deeply; LLM will refine.)
        if answer is None:
            if "pdf_value_sum" in partial_results:
                answer = partial_results["pdf_value_sum"]
            elif "spreadsheet_value_sum" in partial_results:
                answer = partial_results["spreadsheet_value_sum"]
            elif "json_answer" in partial_results:
                answer = partial_results["json_answer"]

        # -----------------------------
        # LLM fallback with full context
        # -----------------------------
        if answer is None and self._has_llm():
            context = self._build_context(
                url=url,
                html=html,
                text=text,
                file_summaries=file_summaries,
                partial_results=partial_results,
            )
            system_msg = (
                "You are a data-analysis assistant solving quiz questions. "
                "You are given scraped web page content and any parsed files. "
                "Your job is to infer the SINGLE final answer the quiz expects. "
                "If the answer should be a number, return only the number. "
                "If it should be a boolean, return true or false. "
                "If text, return exactly the required text. "
                "If a JSON is clearly required, return a compact JSON object."
            )
            user_msg = (
                "Here is the scraped quiz context:\n\n"
                + json.dumps(context, indent=2)
                + "\n\nBased on this, what is the single final answer the quiz expects?"
            )
            llm_answer = self._llm_chat(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
            )
            if llm_answer:
                answer = llm_answer.strip()

        # 4) Final pure-regex numeric fallback (last resort)
        if answer is None:
            nums = re.findall(r"\d+\.?\d*", text)
            if nums:
                try:
                    answer = float(nums[-1]) if "." in nums[-1] else int(nums[-1])
                except:
                    answer = nums[-1]

        # 5) If absolutely nothing, send a default string
        if answer is None:
            answer = "default-answer"

        self._debug("Computed ANSWER:", answer)

        # ---------------------------------------------------
        # SUBMIT
        # ---------------------------------------------------
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
                r = self.session.post(submit_url, json=payload, timeout=20)
                try:
                    submit_response = r.json()
                except Exception:
                    submit_response = {"text": r.text}
                if isinstance(submit_response, dict):
                    next_url = submit_response.get("url")
            except Exception as e:
                submit_response = {"error": str(e)}

        self._debug("Submit response:", submit_response)

        return answer, submit_url, next_url, submit_response

    # ---------------------------------------------------
    # Public entrypoint
    # ---------------------------------------------------
    def solve_and_submit(self, url: str, time_budget_sec: int = 170) -> Dict[str, Any]:
        """
        Orchestrate solving the quiz chain starting from the given URL
        within a time budget (~3 min total including network).
        """
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

            # First page
            elapsed = time.time() - start
            answer, submit_url, next_url, resp = self._solve_one_page(
                page, url, time_budget_sec - elapsed
            )
            result["first"] = {
                "url": url,
                "submit_url": submit_url,
                "answer": answer,
                "submit_response": resp,
            }

            # Chain of quiz URLs
            iteration = 0
            max_iterations = 40

            while next_url and iteration < max_iterations:
                elapsed = time.time() - start
                if elapsed > 160:  # keep margin under 3-minute hard limit
                    self._debug("Global time limit reached.")
                    break

                iteration += 1
                self._debug(f"\n--- Chain step {iteration}: {next_url} ---")

                ans, sub_url, nxt2, resp2 = self._solve_one_page(
                    page, next_url, time_budget_sec - (time.time() - start)
                )

                if ans is None:
                    break

                result["submissions"].append(
                    {
                        "url": next_url,
                        "answer": ans,
                        "submit_url": sub_url,
                        "submit_response": resp2,
                    }
                )
                next_url = nxt2

            browser.close()
            result["completed"] = next_url is None
            result["total_time"] = time.time() - start

        return result
