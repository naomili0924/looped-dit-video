"""Results page of the capability eval: cases file + ldvflux.cap_judge outputs -> <web>/index.html (data inlined into
capage_template.html). Videos/posters are the re-encoded copies in <web>/{clips,gt,poster}.

    python3 build_capage.py <run dir> <cases json> <web dir>
"""
import json, sys
from pathlib import Path
import numpy as np

FAMILY = {"pour": "Pour / scoop", "rotate": "Turn / twist", "tool_use": "Tool use", "press": "Press",
          "locomotion": "Base move / turn", "transport": "Carry / move", "gesture": "Gesture",
          "cut_shape": "Cut / knead", "mix": "Stir / shake", "tie": "Tie / untie", "heldout": "Held-out task"}
TASK_INFO = Path("/workspace/loop_dit_eval/agibot_beta_meta/task_info")


def task_name(task):
    p = TASK_INFO / f"task_{task}.json"
    return json.loads(p.read_text())[0]["task_name"] if p.exists() else None


def nanmean(x):
    x = [v for v in x if v is not None]
    return round(float(np.mean(x)), 4) if x else None


def main():
    run, cases_p, web = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cases = json.loads(cases_p.read_text())["cases"]
    judge = json.loads((run / "judge.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    gt_meta = json.loads((run / "gt" / "meta.json").read_text())
    pick = lambda a: {k: a.get(k) for k in ("completed", "attempted", "plausible", "reason", "score")}
    out = []
    for c in cases:
        cid = c["id"]
        clips, cross = {}, {}
        for pk in ("real", "alt1", "alt2"):
            clips[pk] = []
            for s in (0, 1, 2):
                name = f"{pk}_N1_s{s}"
                m = metrics.get(f"{cid}/{name}", {})
                clips[pk].append({"seed": s, "file": f"clips/{cid}/{name}.mp4", **pick(judge.get(f"{cid}/{name}", {})),
                                  "motion": nanmean(m.get("motion", [])),
                                  "psnr": nanmean(m.get("gt_psnr", [])), "lpips": nanmean(m.get("gt_lpips", []))})
                if f"cross/{cid}/{name}" in judge:
                    cross.setdefault(pk, []).append(pick(judge[f"cross/{cid}/{name}"]))
        out.append({
            "id": cid, "family": c["family"], "familyLabel": FAMILY[c["family"]], "split": c["split"],
            "task": c["task"], "taskName": task_name(c["task"]), "episode": c["episode"], "segFrames": c["segment_frames"],
            "segSec": round(c["segment_frames"] / 30, 1), "caption": c["caption"], "prompts": c["prompts"],
            "provenance": c["provenance"], "gtFrames": gt_meta[cid]["gt_frames"], "gtFull": gt_meta[cid]["gt_full"],
            "gtMotion": nanmean(metrics.get(f"gt/{cid}", {}).get("motion", [])),
            "calib": {k: {**pick(judge[f"gt/{cid}/{k}"]), "prompt": judge[f"gt/{cid}/{k}"]["prompt"]}
                      for k in ("real", "foreign") if f"gt/{cid}/{k}" in judge},
            "clips": clips, "cross": cross})
    data = {"cases": out, "calibration": summary["calibration"]}
    tpl = (Path(__file__).parent / "capage_template.html").read_text()
    (web / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print("wrote", web / "index.html", len(out), "cases")


if __name__ == "__main__":
    main()
