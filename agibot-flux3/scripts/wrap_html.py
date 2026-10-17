#!/usr/bin/env python3
"""Wrap a page fragment (as written for the claude.ai artifact skeleton: <title>, <style>, body content) into a complete
HTML document for GitHub Pages.    scripts/wrap_html.py <in.html> [<out.html>]"""
import re, sys
from pathlib import Path

src = Path(sys.argv[1]); dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
s = src.read_text()
if s.lstrip().lower().startswith("<!doctype"):
    sys.exit(0)
m = re.search(r"<title>(.*?)</title>\s*", s, re.S)
title = m.group(1) if m else src.stem
body = s.replace(m.group(0), "", 1) if m else s
body = re.sub(r'<meta name="claude:viewport-fit-sides"[^>]*>\s*', "", body)
dst.write_text("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
               f"<title>{title}</title>\n<style>:root{{color-scheme:light;box-sizing:border-box}}body{{margin:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>\n"
               "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
print("wrapped", dst)
