"""End-to-end Playwright test of the GeneWeaver Streamlit UI."""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8501"
HERE = Path(__file__).parent
UPLOAD = HERE / "test_upload.fasta"
SHOTS = HERE / "e2e_shots"
SHOTS.mkdir(exist_ok=True)

results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

def shot(page, name):
    page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=False)

def nav(page, label):
    # sidebar radio options
    page.locator("[data-testid='stSidebar'] label", has_text=label).first.click()
    page.wait_for_timeout(1500)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_selector("text=GeneWeaver", timeout=60000)
        page.wait_for_timeout(2000)
        check("app loads (Home)", "CRISPR protospacer" in page.content() or "Genome" in page.content())
        shot(page, "01_home")

        # --- Genome Upload: load example ---
        nav(page, "Genome Upload")
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Load example FASTA").click()
        page.wait_for_timeout(4000)
        content = page.content()
        check("example FASTA loads + stats show", "Sequences" in content and "Total bases" in content)
        shot(page, "02_upload_example")

        # --- Upload own FASTA ---
        page.locator("input[type='file']").set_input_files(str(UPLOAD))
        page.wait_for_timeout(4000)
        content = page.content()
        check("own FASTA parses + stats", "test_upload.fasta" in content
              and "Loaded" in content)
        shot(page, "03_upload_own")

        # --- CRISPR Analysis ---
        nav(page, "CRISPR Analysis")
        page.wait_for_timeout(1000)
        # guide input is inside form; set value
        guide = page.locator("input[aria-label*='Guide'], input[placeholder*='guide' i]").first
        # fallback: first text_input in main
        inputs = page.locator("[data-testid='stMain'] input[type='text'], [data-testid='stMain'] input:not([type])")
        if inputs.count() >= 1:
            inputs.nth(0).fill("ACGTACGTACGTACGTACGTACGTACGTAC")
        if inputs.count() >= 2:
            inputs.nth(1).fill("NGG")
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Analyze").click()
        # wait for success, no-candidates warning, or error
        try:
            page.wait_for_selector(
                "text=/Analysis complete|No candidates found|Analysis failed|Invalid input/",
                timeout=120000)
            ok = True
        except Exception:
            ok = False
        content = page.content()
        good = ok and ("Analysis complete" in content
                       or "No candidates found" in content)
        check("analysis runs to completion", good,
              "" if good else page.locator("[data-testid='stMain']").inner_text()[:300])
        shot(page, "04_analysis")

        # --- Results page ---
        nav(page, "Analysis Results")
        page.wait_for_timeout(1500)
        content = page.content()
        check("results page shows candidates", "Rank" in content or "candidates" in content.lower())
        shot(page, "05_results")

        # --- Performance ---
        nav(page, "Performance")
        page.wait_for_timeout(1500)
        content = page.content()
        check("performance metrics real", "Throughput" in content or "Scan time" in content or "benchmark" in content.lower())
        shot(page, "06_performance")

        # --- Reports: verify download buttons exist ---
        nav(page, "Reports")
        page.wait_for_timeout(1500)
        content = page.content()
        check("reports page has CSV/JSON/HTML", all(s in content for s in ["CSV", "JSON", "HTML"]))
        # actually download each report and verify non-empty content
        for label, ext in [("Download CSV", "csv"), ("Download JSON", "json"),
                           ("Download HTML report", "html")]:
            try:
                with page.expect_download(timeout=15000) as dl_info:
                    page.get_by_role("button", name=label).click()
                dl = dl_info.value
                out = SHOTS / f"download.{ext}"
                dl.save_as(str(out))
                size = out.stat().st_size
                check(f"{label} downloads ({size} bytes)", size > 100)
            except Exception as e:
                check(f"{label} downloads", False, str(e)[:120])
        shot(page, "07_reports")

        # --- Invalid input ---
        nav(page, "CRISPR Analysis")
        page.wait_for_timeout(1000)
        inputs = page.locator("[data-testid='stMain'] input[type='text'], [data-testid='stMain'] input:not([type])")
        if inputs.count() >= 1:
            inputs.nth(0).fill("NOT_A_GUIDE!!!")
        page.get_by_role("button", name="Analyze").click()
        page.wait_for_timeout(3000)
        content = page.content()
        check("invalid guide shows error", "Invalid input" in content or "error" in content.lower())
        shot(page, "08_invalid")

        js_errors = [e for e in errors if "favicon" not in e.lower()]
        check("no JS/page errors", len(js_errors) == 0, "; ".join(js_errors[:3]))
        browser.close()

    fails = [r for r in results if not r[1]]
    print(f"\n=== {len(results)-len(fails)}/{len(results)} checks passed ===")
    sys.exit(1 if fails else 0)

main()
