"""Index an uncompressed tar on the HF Hub without downloading it: walk its headers with HTTP Range requests.

Each member is (name, data offset, size). Only the 512-byte headers (plus whatever small members happen to
fall inside the same read block) are transferred. GNU long names ('L') and pax headers ('x') are handled.

    python scripts/tar_index.py observations/327/684757-685393.tar --out /dev/shm/agibot/index/327_684757-685393.json
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path

import requests

REPO = "agibot-world/AgiBotWorld-Beta"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
BLOCK = 512


def token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok and Path("/workspace/.env").exists():  # tokens normally come from the environment
        for line in Path("/workspace/.env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip()
    return tok


class RangeReader:
    """Range reads over one keep-alive session; re-resolves the signed CDN URL when it expires."""

    def __init__(self, path: str, session: requests.Session | None = None):
        self.path, self.s = path, session or requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token()}"
        self.url, self.size, self.requests, self.bytes = None, None, 0, 0
        self._resolve()

    def _resolve(self) -> None:
        r = self.s.get(BASE + self.path, headers={"Range": "bytes=0-0"}, allow_redirects=False, timeout=60)
        if r.status_code in (301, 302, 303, 307, 308):
            self.url = r.headers["Location"]
        elif r.status_code == 206:
            self.url = BASE + self.path
        else:
            raise RuntimeError(f"{self.path}: HTTP {r.status_code} {r.text[:200]}")
        r = self.s.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60)
        self.size = int(r.headers["Content-Range"].split("/")[-1])

    def read(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        for attempt in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (401, 403):  # signed URL expired
                    self._resolve()
                    continue
                r.raise_for_status()
                self.requests += 1
                self.bytes += len(r.content)
                return r.content
            except requests.RequestException:
                time.sleep(2**attempt)
        raise RuntimeError(f"{self.path}: range {start}-{end} failed")


def walk(reader: RangeReader, read_block: int = 64 * 1024, limit: int | None = None) -> list[dict]:
    """Members in tar order: {"name", "offset" (of the data), "size", "type"}."""
    members, pos = [], 0
    buf, buf_start = b"", 0
    long_name = None
    while pos + BLOCK <= reader.size:
        if not (buf_start <= pos and pos + BLOCK <= buf_start + len(buf)):
            buf, buf_start = reader.read(pos, read_block), pos
        hdr = buf[pos - buf_start : pos - buf_start + BLOCK]
        if hdr == b"\0" * BLOCK:
            break  # end-of-archive marker
        ti = tarfile.TarInfo.frombuf(hdr, tarfile.ENCODING, "surrogateescape")
        data = pos + BLOCK
        padded = (ti.size + BLOCK - 1) // BLOCK * BLOCK
        if ti.type in (tarfile.GNUTYPE_LONGNAME, tarfile.XHDTYPE):
            raw = reader.read(data, ti.size) if not (data + ti.size <= buf_start + len(buf)) else \
                buf[data - buf_start : data - buf_start + ti.size]
            if ti.type == tarfile.GNUTYPE_LONGNAME:
                long_name = raw.rstrip(b"\0").decode("utf-8", "surrogateescape")
            else:  # pax: "len key=value\n" records
                for rec in raw.decode("utf-8", "surrogateescape").split("\n"):
                    if " path=" in rec:
                        long_name = rec.split(" path=", 1)[1]
        else:
            members.append({"name": long_name or ti.name, "offset": data, "size": ti.size,
                            "type": ti.type.decode() if isinstance(ti.type, bytes) else str(ti.type)})
            long_name = None
            if limit and len(members) >= limit:
                break
        pos = data + padded
    return members


# ---------------------------------------------------------------------------
# Episode index: walk only the video headers and binary-search past the depth PNGs
# ---------------------------------------------------------------------------

PROBE_WINDOW = 256 * 1024  # > the largest depth PNG (~150 KB): bisection stops once the bracket is this small
# Growing reads for "first header at or after p": a depth PNG is ~150 KB, so most probes stop within the first
# two reads; 1 MB without a header means the probe sits inside a video.
PROBE_READS = (32 * 1024, 224 * 1024, 768 * 1024)


def parse_header(block: bytes) -> tarfile.TarInfo | None:
    if len(block) < BLOCK or block[257:262] != b"ustar":
        return None
    try:
        return tarfile.TarInfo.frombuf(block, tarfile.ENCODING, "surrogateescape")
    except tarfile.HeaderError:
        return None


def header_at_or_after(reader: RangeReader, p: int) -> tuple[int, str] | None:
    """First valid tar header at a 512-aligned position >= p within ~1 MB: (position, name), or None."""
    p = p // BLOCK * BLOCK
    for length in PROBE_READS:
        if p >= reader.size:
            return None
        buf = reader.read(p, length)
        for i in range(0, len(buf) - BLOCK + 1, BLOCK):
            ti = parse_header(buf[i : i + BLOCK])
            if ti is not None:
                return p + i, ti.name
        p += len(buf)
    return None


def walk_from(reader: RangeReader, pos: int, stop) -> tuple[list[dict], int]:
    """Header walk from `pos` (one 512-byte read per member) until `stop(name)` or end of archive.
    Returns (members walked, position of the stopping header or reader.size)."""
    members, long_name = [], None
    while pos + BLOCK <= reader.size:
        hdr = reader.read(pos, BLOCK)
        if hdr == b"\0" * BLOCK:
            return members, reader.size
        ti = tarfile.TarInfo.frombuf(hdr, tarfile.ENCODING, "surrogateescape")
        data, padded = pos + BLOCK, (ti.size + BLOCK - 1) // BLOCK * BLOCK
        if ti.type == tarfile.GNUTYPE_LONGNAME:
            long_name = reader.read(data, ti.size).rstrip(b"\0").decode("utf-8", "surrogateescape")
        elif ti.type != tarfile.XHDTYPE:
            name = long_name or ti.name
            long_name = None
            if stop(name):
                return members, pos
            members.append({"name": name, "offset": data, "size": ti.size})
        pos = data + padded
    return members, reader.size


def index_episodes(reader: RangeReader, log=None) -> dict[str, dict]:
    """{episode_id: {"videos": {camera: [offset, size]}, "start": pos, "end": pos}} for every episode of the tar.

    Members are grouped by episode (``<ep>/videos/*.mp4`` then ``<ep>/depth/*.png``); the end of an episode's
    depth run is found by bisection on "does the first header at or after p belong to <ep>/depth/".
    """
    episodes: dict[str, dict] = {}
    pos = 0
    while pos < reader.size:
        first = walk_from(reader, pos, lambda n: False if not n else True)  # read the episode's first header
        if first[1] >= reader.size:
            break
        _, ep_start = first
        hdr = reader.read(ep_start, BLOCK)
        ep = tarfile.TarInfo.frombuf(hdr, tarfile.ENCODING, "surrogateescape").name.split("/")[0]
        depth = f"{ep}/depth/"
        # 1) videos and directory entries up to the first depth PNG
        members, depth_start = walk_from(reader, ep_start, lambda n: n.startswith(depth) and n.endswith(".png")
                                         or not n.startswith(f"{ep}/") and n != ep)
        videos = {Path(m["name"]).stem: [m["offset"], m["size"]] for m in members if m["name"].endswith(".mp4")}

        def in_depth(p: int) -> bool:
            h = header_at_or_after(reader, p)
            return h is not None and h[1].startswith(depth)

        end = depth_start
        if depth_start < reader.size and in_depth(depth_start):
            # 2) gallop, then bisect, to the end of the depth run
            lo, step = depth_start, 64 << 20
            hi = min(lo + step, reader.size)
            while hi < reader.size and in_depth(hi):
                lo, step = hi, step * 2
                hi = min(lo + step, reader.size)
            while hi - lo > PROBE_WINDOW:
                mid = (lo + hi) // 2 // BLOCK * BLOCK
                lo, hi = (mid, hi) if in_depth(mid) else (lo, mid)
            start_hdr = header_at_or_after(reader, lo)[0]
            # 3) exact boundary: walk the last few PNGs; videos stored after the depth run are kept too
            tail, end = walk_from(reader, start_hdr, lambda n: not n.startswith(f"{ep}/"))
            videos.update({Path(m["name"]).stem: [m["offset"], m["size"]] for m in tail if m["name"].endswith(".mp4")})
        episodes[ep] = {"videos": videos, "start": ep_start, "end": end}
        if log:
            log(ep, len(episodes), end, reader)
        if end <= pos:
            raise RuntimeError(f"no progress at {pos}")
        pos = end
    return episodes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="path inside the dataset repo, e.g. observations/327/684757-685393.tar")
    p.add_argument("--out", required=True)
    p.add_argument("--members", action="store_true", help="full member walk instead of the episode index (slow)")
    p.add_argument("--limit", type=int, help="--members: stop after this many members (probing)")
    p.add_argument("--read-block", type=int, default=64 * 1024)
    args = p.parse_args()
    t0 = time.time()
    reader = RangeReader(args.path)
    if args.members:
        result = {"members": walk(reader, args.read_block, args.limit), "complete": args.limit is None}
    else:
        result = {"episodes": index_episodes(reader), "complete": True}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"path": args.path, "size": reader.size, **result}))
    os.replace(tmp, out)
    n = len(result.get("episodes", result.get("members", [])))
    print(json.dumps({"path": args.path, "tar_GB": round(reader.size / 1e9, 2), "entries": n,
                      "requests": reader.requests, "transferred_MB": round(reader.bytes / 1e6, 1),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
