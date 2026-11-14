#!/usr/bin/env python3
"""Fail-fast browser smoke test for the deployed Dry Run evidence page.

This intentionally checks the rendered page, not just HTTP status codes.  Run
it after every Fly deploy at both judge viewport sizes:

    python scripts/smoke_public.py --url https://dryrun-junction-2026.fly.dev
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def check_viewport(page, width: int, height: int, screenshot: Path | None) -> dict:
    errors: list[str] = []
    api_errors: list[str] = []

    def on_console(message):
        if message.type == "error":
            errors.append(f"console: {message.text}")

    def on_page_error(error):
        errors.append(f"pageerror: {error}")

    def on_response(response):
        if "/api/" not in response.url:
            return
        if response.status >= 400:
            api_errors.append(f"{response.status} {response.url}")
            return
        try:
            body = response.json()
        except Exception:
            return
        if isinstance(body, dict) and (body.get("status") == "unavailable" or body.get("error")):
            api_errors.append(f"API error in {response.url}: {body.get('error') or body.get('message') or body.get('status')}")

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("response", on_response)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(page.url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_function(
        """() => document.querySelector('#status-text')?.textContent !== 'loading evidence'""",
        timeout=60_000,
    )
    page.wait_for_function(
        """() => document.querySelector('#hero-score')?.textContent.trim() !== 'N/A' &&
                  document.querySelector('#replay-state')?.textContent.startsWith('seed ')""",
        timeout=60_000,
    )
    # Camera videos are evidence, so an empty fallback is a failure.
    page.wait_for_function(
        """() => [...document.querySelectorAll('video')].every(v => v.readyState >= 2 && !v.classList.contains('hide'))""",
        timeout=60_000,
    )
    result = page.evaluate(
        """() => {
          const text = document.body.innerText;
          const bad = [...document.querySelectorAll('#benchmark-metrics strong, #db-health b, #inspector td')]
            .filter(x => x.textContent.trim() === 'N/A').map(x => x.parentElement.innerText);
          const emptyData = [...document.querySelectorAll('#inspector .empty')]
            .filter(x => /No inspector|No evaluation/i.test(x.textContent)).map(x => x.textContent.trim());
          const canvas = document.querySelector('#viewer');
          let nonBackground = 0;
          if (canvas && canvas.width && canvas.height) {
            const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
            if (gl) {
              const pixels = new Uint8Array(canvas.width * canvas.height * 4);
              gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
              for (let i = 0; i < pixels.length; i += 4) {
                if (pixels[i] !== 20 || pixels[i+1] !== 38 || pixels[i+2] !== 60) nonBackground++;
              }
            }
          }
          return {
            score: document.querySelector('#hero-score')?.textContent.trim(),
            replay: document.querySelector('#replay-state')?.textContent.trim(),
            tiles_with_na: bad,
            non_background_pixels: nonBackground,
            scroll_width: document.documentElement.scrollWidth,
            viewport_width: window.innerWidth,
            video_ready: [...document.querySelectorAll('video')].map(v => v.readyState),
            empty_fallbacks: [...document.querySelectorAll('.camera .empty')]
              .filter(x => !x.classList.contains('hide')).map(x => x.textContent.trim()),
            empty_data: emptyData,
          };
        }"""
    )
    if result["tiles_with_na"]:
        errors.append(f"N/A benchmark tiles: {result['tiles_with_na']}")
    if result["non_background_pixels"] < 100:
        errors.append(f"3D viewer is blank ({result['non_background_pixels']} non-background pixels)")
    if result["empty_fallbacks"]:
        errors.append(f"empty camera fallbacks: {result['empty_fallbacks']}")
    if result["empty_data"]:
        errors.append(f"empty data sections: {result['empty_data']}")
    if result["scroll_width"] != result["viewport_width"]:
        errors.append(f"horizontal overflow: scrollWidth={result['scroll_width']} viewport={result['viewport_width']}")
    errors.extend(api_errors)
    if screenshot:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot), full_page=True)
    return {"width": width, "height": height, **result, "errors": errors, "api_errors": api_errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://dryrun-junction-2026.fly.dev/")
    parser.add_argument("--screenshot-dir", type=Path)
    args = parser.parse_args()
    executable = shutil.which("chromium") or shutil.which("google-chrome")
    if not executable:
        raise SystemExit("headless Chromium is required")
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        page = browser.new_page()
        for width, height, name in ((1440, 900, "laptop"), (390, 844, "phone")):
            page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
            screenshot = args.screenshot_dir / f"{name}.png" if args.screenshot_dir else None
            results.append(check_viewport(page, width, height, screenshot))
        browser.close()
    print(json.dumps({"url": args.url, "viewports": results}, indent=2))
    return 1 if any(r["errors"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
