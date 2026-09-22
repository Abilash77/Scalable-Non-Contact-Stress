#!/usr/bin/env python3
"""
Browser test of the real dashboard path (Chrome via Playwright).

Chrome gets a fake camera that plays test_face.jpg (as a .y4m video) and a fake
microphone, so the page's own getUserMedia -> /api/upload_frame and
/api/upload_audio code runs unchanged. Keyboard input is real key events typed
into the textarea; handwriting is real mouse strokes on the canvas.

    PORT=5055 python run.py &
    python scripts/verify_dashboard_browser.py --base http://127.0.0.1:5055
"""
import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(cond, msg):
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def write_y4m(path, img, frames=30):
    img = cv2.resize(img, (640, 480))
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    with open(path, "wb") as f:
        f.write(b"YUV4MPEG2 W640 H480 F15:1 Ip A1:1 C420jpeg\n")
        for _ in range(frames):
            f.write(b"FRAME\n")
            f.write(yuv.tobytes())


def status(base):
    return json.loads(urllib.request.urlopen(base + "/status", timeout=10).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    y4m = os.path.join(tempfile.gettempdir(), "ra_hmsd_face.y4m")
    write_y4m(y4m, cv2.imread(os.path.join(ROOT, "test_face.jpg")))
    text = ("I am working on my project today and typing this sentence at my normal pace "
            "so the keyboard model can see real timing.")

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chrome, headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={y4m}", "--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(permissions=["camera", "microphone"])
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.base + "/")
        check(page.locator("text=HOW THE INPUTS WORK").count() == 1, "HOW THE INPUTS WORK section present")
        page.click("button.btn-start")
        page.wait_for_timeout(6000)

        s = status(args.base)
        check(s["camera_connected"] and s["camera_frame_count"] > 0, f"browser camera frames received ({s['camera_frame_count']})")
        check(s["face_detected"] and s["face_bbox"] is not None, f"FACE DETECTED via browser camera, bbox={s['face_bbox']}")
        check(s["eye_status"] == "DETECTED", f"EYES DETECTED via browser camera ({s['eye_status']})")
        check(page.locator("#face-box").is_visible() and "FACE DETECTED" in page.inner_text("#face-box-label"),
              "dashboard face box overlay shows FACE DETECTED")
        check(s["mic_status"] == "RECEIVING_AUDIO" and s["audio_sample_count"] > 0,
              f"browser microphone -> RECEIVING_AUDIO ({s['audio_sample_count']} chunks)")

        # Keyboard: real key events at a human-like pace
        page.click("#keyboard-input")
        rng = np.random.RandomState(0)
        for ch in text:
            page.keyboard.type(ch, delay=int(rng.uniform(60, 140)))
            page.wait_for_timeout(int(rng.uniform(40, 220)))
        s = None
        for _ in range(40):
            s = status(args.base)
            if s["prediction"] in ("STRESS", "NON-STRESS"):
                break
            time.sleep(0.25)
        check(s["prediction"] in ("STRESS", "NON-STRESS") and s["stress_probability"] is not None,
              f"typing -> prediction {s['prediction']} P(stress)={s['stress_probability']} "
              f"({s['keyboard_event_count']} keys, window {s['buffer_fills'].get('keystroke')}/10)")
        check(s["latest_features"]["keyboard_7d"] is not None,
              f"live keyboard 7D features {s['latest_features']['keyboard_7d']}")
        page.wait_for_timeout(1200)
        state = page.inner_text("#current-state")
        check(state in ("STRESS", "NON-STRESS"), f"dashboard CURRENT STATE shows {state}")
        check("10/10" in page.inner_text("#keyboard-sample-progress"), page.inner_text("#keyboard-sample-progress"))

        # Handwriting: real mouse strokes
        box = page.locator("#hw-canvas").bounding_box()
        for k in range(3):
            page.mouse.move(box["x"] + 40 + 150 * k, box["y"] + 50)
            page.mouse.down()
            for i in range(25):
                page.mouse.move(box["x"] + 40 + 150 * k + 4 * i, box["y"] + 50 + 30 * np.sin(i / 3), steps=2)
            page.mouse.up()
        prev = status(args.base)["handwriting_submission_count"]
        page.click(".handwriting-actions .btn-start")
        for _ in range(30):
            s = status(args.base)
            if s["handwriting_submission_count"] > prev:
                break
            time.sleep(0.2)
        page.wait_for_timeout(800)
        s = status(args.base)
        hw = s["latest_features"]["handwriting_9d"]
        check(s["handwriting_submission_count"] > prev and hw is not None and hw[7] > 0,
              f"canvas strokes -> 9D handwriting features {hw}")
        check(s["runtime_modality_status"]["handwriting"]["status"] == "ACTIVE", "handwriting ACTIVE after submit")

        panel = page.inner_text("#reliability-panel")
        check("STRESS MODEL: LOADED" in panel and panel.count("NOT STRESS-TRAINED") == 4,
              "keyboard = stress model; speech/facial/eye/handwriting = NOT STRESS-TRAINED")
        page.screenshot(path=os.path.join(tempfile.gettempdir(), "ra_hmsd_dashboard.png"), full_page=True)
        check(not errors, f"no JavaScript errors {errors[:3]}")
        page.click("button.btn-stop")
        browser.close()

    print(f"\n{'ALL BROWSER CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
