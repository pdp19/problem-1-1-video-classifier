from __future__ import annotations

import html
import json
import os
from pathlib import Path

from .. import taxonomy

GOLD_FIELDS = ("video_id", "label", "secondary", "draft_label", "draft_confidence", "changed", "note")

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Review</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e2e1dc; --accent:#2457d6; --warn:#b4560b; --ok:#2f7d32; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --card:#1f1f1d; --ink:#ecebe6; --muted:#a09f99; --line:#33332f; --accent:#7aa2ff; --warn:#f0a35b; --ok:#7fc683; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system, system-ui, sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:16px; margin:0 12px 0 0; }
.stat { color:var(--muted); }
button, select, input[type=text] { font:inherit; color:inherit; background:var(--card); border:1px solid var(--line); border-radius:6px; padding:5px 8px; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); cursor:pointer; }
main { padding:16px; display:grid; gap:16px; max-width:1500px; margin:0 auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; display:grid; grid-template-columns:minmax(0,1.6fr) minmax(0,1fr); overflow:hidden; }
.card.done { border-color:var(--ok); }
.card img { width:100%; display:block; cursor:zoom-in; }
.meta { padding:12px 14px; display:flex; flex-direction:column; gap:8px; }
.vid { font-weight:600; font-family:ui-monospace, monospace; }
.conf { font-weight:600; }
.conf.low { color:var(--warn); }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.row label { color:var(--muted); min-width:72px; }
.small { color:var(--muted); font-size:12.5px; }
blockquote { margin:0; padding-left:10px; border-left:3px solid var(--line); color:var(--muted); font-size:12.5px; }
.changed { color:var(--warn); font-weight:600; }
dialog { padding:0; border:none; max-width:96vw; background:transparent; }
dialog img { max-width:96vw; max-height:92vh; }
details summary { cursor:pointer; color:var(--muted); }
@media (max-width: 820px) { .card { grid-template-columns:1fr; } }
</style></head><body>
<header>
  <h1>Label Review</h1>
  <span class="stat" id="progress"></span>
  <select id="filter"><option value="">All classes</option>__OPTIONS__</select>
  <select id="sort"><option value="conf">Lowest confidence first</option><option value="id">By video id</option></select>
  <label class="stat"><input type="checkbox" id="hideDone"> hide reviewed</label>
  <button class="primary" id="download">Download labels.csv</button>
  <span class="stat" id="counts"></span>
</header>
<main id="cards"></main>
<dialog id="zoom"><img alt=""></dialog>
<script>
const DATA = __DATA__;
const CLASSES = __CLASSES__;
const DEFS = __DEFS__;
const KEY = "vclf-review-v1";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function cur(d) { return Object.assign({label:d.label, secondary:d.secondary, note:"", done:false}, state[d.video_id] || {}); }
function opts(sel, withNone) {
  const list = withNone ? ["none", ...CLASSES] : CLASSES;
  return list.map(c => `<option value="${c}" ${c===sel?"selected":""} title="${DEFS[c]||""}">${c}</option>`).join("");
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function render() {
  const f = document.getElementById("filter").value;
  const sort = document.getElementById("sort").value;
  const hide = document.getElementById("hideDone").checked;
  let rows = DATA.slice();
  rows.sort(sort === "conf" ? (a,b) => a.confidence - b.confidence : (a,b) => a.video_id.localeCompare(b.video_id));
  rows = rows.filter(d => (!f || cur(d).label === f) && (!hide || !cur(d).done));
  document.getElementById("cards").innerHTML = rows.map(d => {
    const c = cur(d);
    const changed = c.label !== d.label;
    return `<article class="card ${c.done?"done":""}" data-id="${d.video_id}">
      <img src="${d.sheet}" alt="contact sheet ${d.video_id}" loading="lazy">
      <div class="meta">
        <div class="row"><span class="vid">${d.video_id}</span>
          <span class="small">${Math.round(d.duration_s)} s · ${esc(d.domain)} / ${esc(d.sub_category)}</span></div>
        <div class="row"><label>Draft</label><b>${d.label}</b>
          <span class="conf ${d.confidence < 0.7 ? "low" : ""}">${Math.round(d.confidence*100)}%</span>
          <span class="small">2nd: ${d.secondary}</span></div>
        <div class="small">${esc(d.rationale)}</div>
        <div class="row"><label>Label</label><select data-k="label">${opts(c.label,false)}</select>
          ${changed ? '<span class="changed">changed</span>' : ""}</div>
        <div class="row"><label>Secondary</label><select data-k="secondary">${opts(c.secondary,true)}</select></div>
        <div class="row"><label>Note</label><input type="text" data-k="note" value="${esc(c.note)}" placeholder="optional"></div>
        <div class="row"><label></label><label><input type="checkbox" data-k="done" ${c.done?"checked":""}> reviewed</label></div>
        ${d.excerpts.length ? `<details><summary>Transcript excerpts</summary>${d.excerpts.map(e => `<blockquote>${esc(e)}</blockquote>`).join("")}</details>` : '<span class="small">No subtitles</span>'}
      </div></article>`;
  }).join("");
  const done = DATA.filter(d => cur(d).done).length;
  const changedN = DATA.filter(d => cur(d).label !== d.label).length;
  document.getElementById("progress").textContent = `${done}/${DATA.length} reviewed · ${changedN} changed`;
  const counts = {};
  DATA.forEach(d => { const l = cur(d).label; counts[l] = (counts[l]||0) + 1; });
  document.getElementById("counts").textContent = CLASSES.map(c => `${c} ${counts[c]||0}`).join(" · ");
}
document.getElementById("cards").addEventListener("change", e => {
  const card = e.target.closest(".card"); const k = e.target.dataset.k; if (!card || !k) return;
  const d = DATA.find(x => x.video_id === card.dataset.id);
  const s = cur(d); s[k] = e.target.type === "checkbox" ? e.target.checked : e.target.value;
  if (k !== "done" && k !== "note") s.done = true;
  state[d.video_id] = s; save(); render();
});
document.getElementById("cards").addEventListener("click", e => {
  if (e.target.tagName === "IMG") { const z = document.getElementById("zoom"); z.querySelector("img").src = e.target.src; z.showModal(); }
});
document.getElementById("zoom").addEventListener("click", e => e.currentTarget.close());
["filter","sort","hideDone"].forEach(id => document.getElementById(id).addEventListener("change", render));
document.getElementById("download").addEventListener("click", () => {
  const q = v => /[",\\n]/.test(v) ? '"' + String(v).replace(/"/g,'""') + '"' : v;
  const lines = [__FIELDS__.join(",")];
  DATA.slice().sort((a,b) => a.video_id.localeCompare(b.video_id)).forEach(d => {
    const c = cur(d);
    lines.push([d.video_id, c.label, c.secondary, d.label, d.confidence, c.label !== d.label ? 1 : 0, c.note].map(q).join(","));
  });
  const blob = new Blob([lines.join("\\n") + "\\n"], {type:"text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "labels.csv"; a.click();
});
render();
</script></body></html>
"""


def build_review_page(draft_rows: dict[str, dict], excerpts: dict[str, list[str]], sheets_dir: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for vid in sorted(draft_rows):
        r = draft_rows[vid]
        sheet = Path(os.path.relpath(sheets_dir / f"{vid}.jpg", out_path.parent)).as_posix()
        data.append({
            "video_id": vid,
            "label": r["label"],
            "secondary": r.get("secondary") or "none",
            "confidence": float(r.get("confidence") or 0),
            "rationale": r.get("rationale", ""),
            "domain": r.get("domain", ""),
            "sub_category": r.get("sub_category", ""),
            "duration_s": float(r.get("duration_s") or 0),
            "excerpts": excerpts.get(vid, []),
            "sheet": sheet,
        })
    options = "".join(f'<option value="{c}">{html.escape(c)}</option>' for c in taxonomy.CLASS_NAMES)
    defs = {c.name: c.definition for c in taxonomy.CLASSES}
    page = (
        _PAGE.replace("__OPTIONS__", options)
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
        .replace("__CLASSES__", json.dumps(list(taxonomy.CLASS_NAMES)))
        .replace("__DEFS__", json.dumps(defs).replace("</", "<\\/"))
        .replace("__FIELDS__", json.dumps(list(GOLD_FIELDS)))
    )
    out_path.write_text(page, encoding="utf-8")
    return out_path
