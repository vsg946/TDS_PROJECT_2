import os
import re
import time
import json
import base64
import tempfile
import requests
import pandas as pd
import pdfplumber
from playwright.sync_api import sync_playwright

class QuizSolver:
    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret

    # -----------------------------
    # Utility
    # -----------------------------

    def _debug(self, *args):
        print("[DEBUG]", *args)

    def _find_submit_url(self, html_text: str):
        """
        Extract a clean submit URL without trailing HTML tags.
        """
        pattern = r"https?://[a-zA-Z0-9\.\-_/]+/submit"
        match = re.search(pattern, html_text)
        return match.group(0) if match else None

    def _extract_atob(self, html_text: str):
        """
        Extract atob(`...`) / atob("...") / atob('...') contents.
        """
        chunks = []
        for m in re.finditer(
            r'atob\(\s*`([^`]*)`\s*\)|atob\(\s*"([^"]*)"\s*\)|atob\(\s*\'([^\']*)\'\s*\)',
            html_text,
            re.DOTALL
        ):
            payload = next((g for g in m.groups() if g), None)
            if payload:
                chunks.append(payload)
        return chunks

    def _decode_base64(self, s: str):
        try:
            return base64.b64decode(s).decode("utf-8", errors="ignore")
        except:
            return None

    def _download_file(self, href):
        if not href.lower().startswith("http"):
            return None
        try:
            r = requests.get(href, timeout=30)
            if r.status_code != 200:
                return None
        except:
            return None

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(r.content)
        tmp.close()
        return tmp.name

    # -----------------------------
    # File Processing
    # -----------------------------

    def _sum_value_in_pdf_page2(self, pdf_path):
        self._debug("Extracting PDF:", pdf_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) < 2:
                    return None

                tables = pdf.pages[1].extract_tables()
                if not tables:
                    return None

                for table in tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    for col in df.columns:
                        if col.strip().lower() == "value":
                            series = pd.to_numeric(df[col], errors="coerce")
                            total = float(series.sum())
                            self._debug("PDF sum:", total)
                            return total
        except Exception as e:
            self._debug("PDF error:", e)

        return None

    def _process_csv_xlsx(self, file_path):
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            for col in df.columns:
                if col.lower() == "value":
                    return float(pd.to_numeric(df[col], errors="coerce").sum())
        except Exception as e:
            self._debug("Spreadsheet parse error:", e)

        return None

    # -----------------------------
    # Solve one page
    # -----------------------------

    def _solve_one_page(self, page, url, time_left):

        self._debug("Visiting:", url)

        # WAIT for JS-rendering
        page.goto(url, wait_until="networkidle", timeout=120000)

        html = page.content()
        submit_url = self._find_submit_url(html)
        self._debug("Submit URL:", submit_url)

        # Extract base64 atob() blocks
        decoded_text_chunks = []
        for raw in self._extract_atob(html):
            decoded = self._decode_base64(raw)
            if decoded:
                decoded_text_chunks.append(decoded)

        # Detect downloadable file
        file_path = None
        for a in page.query_selector_all("a"):
            href = a.get_attribute("href") or ""
            if any(href.lower().endswith(ext) for ext in [".pdf", ".csv", ".xlsx", ".json"]):
                fp = self._download_file(href)
                if fp:
                    self._debug("Downloaded file:", fp)
                    file_path = fp
                    break

        # Determine the answer
        answer = None

        # 1) JSON inside atob
        for txt in decoded_text_chunks:
            try:
                j = json.loads(txt)
                if "answer" in j:
                    answer = j["answer"]
                    break
            except:
                pass

        # 2) If PDF
        if answer is None and file_path and file_path.endswith(".pdf"):
            ans = self._sum_value_in_pdf_page2(file_path)
            if ans is not None:
                answer = ans

        # 3) If CSV/XLSX
        if answer is None and file_path and file_path.endswith((".csv", ".xlsx")):
            ans = self._process_csv_xlsx(file_path)
            if ans is not None:
                answer = ans

        # 4) Fallback: last number on the page
        if answer is None:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", html)
            if nums:
                val = nums[-1]
                try:
                    answer = float(val) if "." in val else int(val)
                except:
                    answer = val

        self._debug("ANSWER:", answer)

        # Submit answer
        submit_resp_obj = None

        if submit_url:
            payload = {
                "email": self.email,
                "secret": self.secret,
                "url": url,
                "answer": answer
            }

            try:
                r = requests.post(submit_url, json=payload, timeout=30)
                try:
                    submit_resp_obj = r.json()
                except:
                    submit_resp_obj = {"text": r.text}
            except Exception as e:
                submit_resp_obj = {"error": str(e)}

        self._debug("SUBMIT RESPONSE:", submit_resp_obj)

        next_url = submit_resp_obj.get("url") if isinstance(submit_resp_obj, dict) else None

        return answer, submit_url, next_url, submit_resp_obj

    # -----------------------------
    # Orchestrator
    # -----------------------------

    def solve_and_submit(self, url, time_budget_sec=170):

        start = time.time()

        result = {
            "first": None,
            "submissions": []
        }

        with sync_playwright() as p:

            # Stable Chromium launch
            for attempt in range(3):
                try:
                    browser = p.chromium.launch(headless=True)
                    break
                except:
                    time.sleep(2)
            page = browser.new_page()

            # First page
            answer, submit_url, next_url, resp = self._solve_one_page(
                page, url, time_budget_sec
            )
            result["first"] = {
                "url": url,
                "submit_url": submit_url,
                "answer": answer,
                "submit_response": resp
            }

            # Quiz chain
            while next_url and (time.time() - start < time_budget_sec):
                ans, s_url, n_url, r = self._solve_one_page(
                    page, next_url, time_budget_sec
                )
                result["submissions"].append({
                    "url": next_url,
                    "answer": ans,
                    "submit_url": s_url,
                    "submit_response": r
                })
                next_url = n_url

            browser.close()

        return result
