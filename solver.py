import os, re, time, json, base64, tempfile, hashlib
import requests, pandas as pd, pdfplumber
from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright


class QuizSolver:
    def __init__(self, email: str, secret: str):
        self.email = email
        self.secret = secret
        self.visited = set()
        self.session = requests.Session()

    # --------------------- UTILITIES ---------------------

    def _debug(self, *a):
        print("[DEBUG]", *a)

    def _find_submit_url(self, html, url):
        m = re.search(r"https?://[^\s\"'<>]+/submit", html)
        if m:
            return m.group(0)

        if "/submit" in html:
            p = urlparse(url)
            return f"{p.scheme}://{p.netloc}/submit"

        return None

    def _decode_base64(self, s):
        try:
            s = s.replace("-", "+").replace("_", "/")
            pad = len(s) % 4
            if pad: s += "=" * (4 - pad)
            return base64.b64decode(s).decode("utf-8", "ignore")
        except:
            return None

    def _extract_atob(self, html):
        out = []
        for m in re.finditer(r'atob\([`"\']([^`"\']+)[`"\']\)', html):
            dec = self._decode_base64(m.group(1))
            if dec:
                out.append(dec)
                if len(out) >= 3:
                    break
        return out

    def _download_file(self, href, base):
        if not href:
            return None
        if not href.lower().startswith("http"):
            href = urljoin(base, href)

        try:
            r = self.session.get(href, timeout=10)
            if r.status_code != 200:
                return None
        except:
            return None

        ext = ""
        ct = r.headers.get("content-type", "").lower()
        if "pdf" in ct or href.endswith(".pdf"): ext = ".pdf"
        elif "csv" in ct or href.endswith(".csv"): ext = ".csv"
        elif "xlsx" in ct or href.endswith(".xlsx"): ext = ".xlsx"
        elif "json" in ct or href.endswith(".json"): ext = ".json"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(r.content)
        tmp.close()
        return tmp.name

    # --------------------- FILE PROCESSING ---------------------

    def _sum_pdf(self, fp):
        try:
            with pdfplumber.open(fp) as pdf:
                total = 0
                found = False
                for page in pdf.pages[:10]:
                    tables = page.extract_tables()
                    if not tables: continue
                    for t in tables:
                        hdr = [str(h).lower() for h in t[0]]
                        if "value" in hdr:
                            idx = hdr.index("value")
                            for row in t[1:]:
                                try:
                                    total += float(row[idx])
                                    found = True
                                except:
                                    pass
                return total if found else None
        except:
            return None

    def _sum_csv_xlsx(self, fp):
        try:
            if fp.endswith(".csv"):
                df = pd.read_csv(fp, nrows=10000)
            else:
                df = pd.read_excel(fp, nrows=10000)
            cols = [c.lower().strip() for c in df.columns]
            for k in ["value", "amount", "total", "sum"]:
                if k in cols:
                    idx = cols.index(k)
                    return float(pd.to_numeric(df.iloc[:, idx], errors="coerce").sum())
        except:
            pass
        return None

    def _json_answer(self, fp):
        try:
            with open(fp) as f:
                j = json.load(f)
            if "answer" in j: return j["answer"]
            if "values" in j and isinstance(j["values"], list):
                return sum(j["values"])
        except:
            pass
        return None

    # --------------------- ANSWER EXTRACTION ---------------------

    def _extract_answer(self, page, html):
        # Try JS first (fast + reliable)
        try:
            ans = page.evaluate("""
                () => {
                    if (window.answer !== undefined) return window.answer;
                    if (window.solution !== undefined) return window.solution;
                    if (window.result !== undefined) return window.result;

                    let el = document.querySelector('[data-answer], #answer, .answer');
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

        # Regex fallback
        m = re.search(r'answer["\s:=]+(\d+\.?\d*)', html, re.I)
        if m:
            try: return float(m.group(1))
            except: return m.group(1)

        return None

    # --------------------- SOLVE ONE PAGE ---------------------

    def _solve_one(self, page, url, time_left):

        if time_left < 5:
            return None, None, None, {"error": "timeout"}

        h = hashlib.md5(url.encode()).hexdigest()
        if h in self.visited:
            return None, None, None, {"error": "visited"}
        self.visited.add(h)

        try:
            page.goto(url, timeout=25000, wait_until="domcontentloaded")
            page.wait_for_timeout(400)
        except:
            pass

        html = page.content()

        submit = self._find_submit_url(html, url)

        atobs = self._extract_atob(html)

        fp = None
        m = re.search(r'href=["\']([^"\']+\.(?:pdf|csv|xlsx|json))["\']', html, re.I)
        if m:
            fp = self._download_file(m.group(1), url)

        # ---- Determine answer ----
        ans = self._extract_answer(page, html)

        if ans is None:
            for chunk in atobs:
                try:
                    j = json.loads(chunk)
                    if "answer" in j:
                        ans = j["answer"]
                        break
                except:
                    pass

        if ans is None and fp:
            if fp.endswith(".pdf"):
                ans = self._sum_pdf(fp)
            elif fp.endswith((".csv", ".xlsx")):
                ans = self._sum_csv_xlsx(fp)
            elif fp.endswith(".json"):
                ans = self._json_answer(fp)

        if ans is None:
            nums = re.findall(r'\d+\.?\d*', html)
            if nums:
                try: ans = float(nums[-1])
                except: ans = nums[-1]

        if ans is None:
            ans = "default-answer"

        # ---- Submit ----
        resp = {"error": "no-submit"}
        if submit:
            payload = {
                "email": self.email,
                "secret": self.secret,
                "url": url,
                "answer": ans
            }
            try:
                r = self.session.post(submit, json=payload, timeout=10)
                try: resp = r.json()
                except: resp = {"text": r.text}
            except:
                resp = {"error": "submit-failed"}

        next_url = resp.get("url") if isinstance(resp, dict) else None
        return ans, submit, next_url, resp

    # --------------------- ORCHESTRATOR ---------------------

    def solve_and_submit(self, url, time_budget_sec=170):

        start = time.time()
        result = {"first": None, "submissions": [], "completed": False}

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
                )
            except Exception as e:
                return {"error": f"browser-fail: {e}"}

            page = browser.new_page()
            page.set_default_timeout(25000)

            ans, submit, nxt, resp = self._solve_one(page, url, time_budget_sec)
            result["first"] = {
                "url": url,
                "submit_url": submit,
                "answer": ans,
                "submit_response": resp
            }

            count = 0
            while nxt and (time.time() - start < 160) and count < 40:
                count += 1
                ans, submit, nxt2, resp = self._solve_one(
                    page, nxt, time_budget_sec - (time.time() - start)
                )
                result["submissions"].append({
                    "url": nxt,
                    "answer": ans,
                    "submit_url": submit,
                    "submit_response": resp
                })
                nxt = nxt2

            browser.close()

            result["completed"] = nxt is None
            result["total_time"] = time.time() - start

        return result
