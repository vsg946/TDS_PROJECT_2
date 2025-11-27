import os
import re
import time
import json
import base64
import tempfile
import requests
import pandas as pd
import pdfplumber
from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import hashlib
import speech_recognition as sr
from pydub import AudioSegment

class QuizSolver:
    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.visited_urls = set()
        self.session = requests.Session()

    def _debug(self, *args):
        print("[DEBUG]", *args)

    # ------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------

    def _find_submit_url(self, html, base_url):
        """Extract /submit endpoint from page."""
        m = re.search(r"https?://[^\s\"'<>]+/submit", html)
        if m:
            return m.group(0)

        if "/submit" in html:
            u = urlparse(base_url)
            return f"{u.scheme}://{u.netloc}/submit"

        return None

    def _decode_base64(self, s):
        try:
            s = s.replace("-", "+").replace("_", "/")
            pad = len(s) % 4
            if pad:
                s += "=" * (4 - pad)
            return base64.b64decode(s).decode("utf-8", errors="ignore")
        except:
            return None

    def _extract_encoded_chunks(self, html):
        chunks = []

        for m in re.finditer(r'atob\(["\']([^"\']+)["\']\)', html):
            d = self._decode_base64(m.group(1))
            if d:
                chunks.append(d)
                if len(chunks) >= 5:
                    break

        return chunks

    def _download_file(self, href, page_url):
        if not href:
            return None

        if not href.startswith("http"):
            href = urljoin(page_url, href)

        try:
            r = self.session.get(href, timeout=12, stream=True)
            if r.status_code != 200:
                return None

            content = b""
            for chunk in r.iter_content(8192):
                content += chunk
                if len(content) > 12 * 1024 * 1024:
                    return None

        except:
            return None

        ext = ""
        ct = r.headers.get("content-type", "").lower()

        if "pdf" in ct or href.endswith(".pdf"):
            ext = ".pdf"
        elif "csv" in ct or href.endswith(".csv"):
            ext = ".csv"
        elif "xlsx" in ct or href.endswith(".xlsx"):
            ext = ".xlsx"
        elif "json" in ct or href.endswith(".json"):
            ext = ".json"
        elif any(href.endswith(e) for e in [".mp3", ".wav", ".ogg", ".m4a", ".flac"]):
            ext = os.path.splitext(href)[1]

        if ext == "":
            return None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(content)
        tmp.close()
        return tmp.name

    # ------------------------------------------------------------
    # FILE PROCESSING
    # ------------------------------------------------------------

    def _process_pdf(self, path):
        try:
            with pdfplumber.open(path) as pdf:
                total = 0
                found = False
                pages = min(len(pdf.pages), 10)

                for i in range(pages):
                    tbls = pdf.pages[i].extract_tables()
                    if not tbls:
                        continue
                    for t in tbls:
                        if len(t) < 2: 
                            continue
                        header = [h.lower().strip() if h else "" for h in t[0]]
                        idx = -1
                        for j,h in enumerate(header):
                            if h in ["value", "amount", "total", "sum"]:
                                idx = j
                                break
                        if idx == -1: 
                            continue
                        for row in t[1:]:
                            try:
                                total += float(row[idx])
                                found = True
                            except:
                                pass
                if found:
                    return total
        except:
            pass
        return None

    def _process_csv_xlsx(self, path):
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            cols = [c.lower().strip() for c in df.columns]
            for pref in ["value", "amount", "total", "sum"]:
                if pref in cols:
                    idx = cols.index(pref)
                    return float(pd.to_numeric(df.iloc[:, idx], errors="coerce").sum())

            nums = df.select_dtypes("number")
            if len(nums.columns):
                return float(nums.sum().sum())
        except:
            return None
        return None

    def _process_json(self, path):
        try:
            d = json.load(open(path))
            if isinstance(d, dict):
                if "answer" in d:
                    return d["answer"]
                if "values" in d and isinstance(d["values"], list):
                    return sum(d["values"])
                return sum(v for v in d.values() if isinstance(v,(int,float)))
        except:
            pass
        return None

    # ------------------------------------------------------------
    # AUDIO PROCESSING
    # ------------------------------------------------------------

    def _transcribe_audio(self, path):
        """Google Speech API free tier (local)."""
        try:
            audio = path
            if not path.endswith(".wav"):
                seg = AudioSegment.from_file(path)
                audio = path.replace(os.path.splitext(path)[1], ".wav")
                seg.export(audio, format="wav")

            rec = sr.Recognizer()
            with sr.AudioFile(audio) as src:
                data = rec.record(src)
                return rec.recognize_google(data)
        except Exception as e:
            self._debug("Audio error", e)
            return None

    def _execute_audio_instructions(self, text, page, url):
        if not text:
            return None

        low = text.lower()

        # detect type
        if "csv" in low:
            t="csv"
        elif "excel" in low or "xlsx" in low:
            t="xlsx"
        elif "pdf" in low:
            t="pdf"
        elif "json" in low:
            t="json"
        else:
            return None

        html = page.content()
        m = re.search(rf'href=["\']([^"\']+\.{t})["\']', html)
        if not m:
            return None

        path = self._download_file(m.group(1), url)
        if not path: return None

        # detect operation
        if any(k in low for k in ["sum","add","total"]):
            op="sum"
        elif any(k in low for k in ["average","mean"]):
            op="avg"
        elif any(k in low for k in ["max","highest"]):
            op="max"
        elif any(k in low for k in ["min","lowest"]):
            op="min"
        elif any(k in low for k in ["count","number of"]):
            op="count"
        else:
            op="sum"

        # load df
        try:
            if t=="csv":
                df = pd.read_csv(path)
            elif t=="xlsx":
                df = pd.read_excel(path)
            elif t=="pdf":
                return self._process_pdf(path)
            elif t=="json":
                return self._process_json(path)
        except:
            return None

        nums = df.select_dtypes("number")
        if not len(nums.columns):
            return None

        col = nums.columns[0]

        if op=="sum":
            return float(nums[col].sum())
        if op=="avg":
            return float(nums[col].mean())
        if op=="max":
            return float(nums[col].max())
        if op=="min":
            return float(nums[col].min())
        if op=="count":
            return int(nums[col].count())

        return None

    # ------------------------------------------------------------
    # ANSWER EXTRACTION
    # ------------------------------------------------------------

    def _extract_answer_from_page(self, page, html):
        try:
            ans = page.evaluate("""
            () => {
                if (window.answer !== undefined) return window.answer;
                if (window.solution !== undefined) return window.solution;
                if (window.result !== undefined) return window.result;

                let el = document.querySelector("[data-answer], .answer, #answer");
                if (el) {
                    if (el.dataset.answer) return el.dataset.answer;
                    return el.textContent.trim();
                }
                return null;
            }
            """)
            if ans is not None:
                try: return float(ans)
                except: return ans
        except:
            pass

        m = re.search(r'answer["\s:=]+(\d+\.?\d*)', html)
        if m:
            return float(m.group(1))

        return None

    # ------------------------------------------------------------
    # Solve a single page
    # ------------------------------------------------------------

    def _solve_one(self, page, url, budget):
        if budget < 5:
            return None,None,None,{"error":"timeout"}

        h = hashlib.md5(url.encode()).hexdigest()
        if h in self.visited_urls:
            return None,None,None,{"error":"visited"}

        self.visited_urls.add(h)
        self._debug("Visiting:", url)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
            page.wait_for_timeout(400)
        except:
            pass

        html = page.content()
        submit = self._find_submit_url(html, url)

        # audio?
        audio_path=None
        am = re.search(r'href=["\']([^"\']+\.(?:mp3|wav|ogg|m4a|flac))["\']', html)
        if am:
            audio_path = self._download_file(am.group(1), url)

        # data file?
        file_path=None
        fm = re.search(r'href=["\']([^"\']+\.(?:pdf|csv|xlsx|json))["\']', html)
        if fm:
            file_path = self._download_file(fm.group(1), url)

        # encoded chunks
        chunks = self._extract_encoded_chunks(html)

        # ---- answer resolution order ----
        answer=None

        # AUDIO
        if not answer and audio_path:
            text = self._transcribe_audio(audio_path)
            answer = self._execute_audio_instructions(text, page, url)

        # DIRECT PAGE
        if answer is None:
            answer = self._extract_answer_from_page(page, html)

        # DECODED JSON
        if answer is None:
            for c in chunks:
                try:
                    j=json.loads(c)
                    if "answer" in j:
                        answer = j["answer"]
                        break
                except:
                    pass

        # FILE
        if answer is None and file_path:
            if file_path.endswith(".pdf"):
                answer = self._process_pdf(file_path)
            elif file_path.endswith(".csv") or file_path.endswith(".xlsx"):
                answer = self._process_csv_xlsx(file_path)
            elif file_path.endswith(".json"):
                answer = self._process_json(file_path)

        # FALLBACK numeric
        if answer is None:
            nums = re.findall(r'\d+\.?\d*', html)
            if nums:
                try: answer = float(nums[-1])
                except: answer = nums[-1]

        # FINAL fallback
        if answer is None:
            answer = "default"

        self._debug("ANSWER:", answer)

        # ------------------------------------------------------------
        # SUBMIT
        # ------------------------------------------------------------

        resp_obj={"error":"no-submit"}
        next_url=None

        if submit:
            payload={
                "email": self.email,
                "secret": self.secret,
                "url": url,
                "answer": answer
            }
            try:
                r=self.session.post(submit,json=payload,timeout=12)
                try: resp_obj = r.json()
                except: resp_obj={"text":r.text}
            except:
                resp_obj={"error":"post-fail"}

            if isinstance(resp_obj,dict):
                next_url = resp_obj.get("url")

        return answer,submit,next_url,resp_obj

    # ------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------

    def solve_and_submit(self, url, time_budget_sec=170):

        start=time.time()
        result={"first":None,"submissions":[],"completed":False}

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox","--disable-setuid-sandbox","--disable-dev-shm-usage"]
            )
            page = browser.new_page()
            page.set_default_timeout(25000)

            # first
            answer,sub,next_url,resp = self._solve_one(page,url,time_budget_sec)
            result["first"]={
                "url":url,
                "submit_url":sub,
                "answer":answer,
                "submit_response":resp
            }

            # chain
            count=0
            while next_url and count<50:
                elapsed=time.time()-start
                if elapsed>165:
                    break
                count+=1
                a,s,n,r=self._solve_one(page,next_url,time_budget_sec-elapsed)
                if a is None:
                    break
                result["submissions"].append({
                    "url":next_url,
                    "answer":a,
                    "submit_url":s,
                    "submit_response":r
                })
                next_url=n

            browser.close()
            result["completed"] = next_url is None
            result["total_time"]=time.time()-start

        return result
