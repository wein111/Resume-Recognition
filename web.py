import io
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

import run_pipeline as rp
from job_ingestion.resume_job_match import rank_jobs_by_skill_overlap
from ner.server.utils import (
    idx2tag,
    merge_entities_by_offsets,
    predict,
    preprocess_data,
)
from skill_normalizer.skill_normalizer.from_ner import skills_for_normalizer


def _analyze_pdf_bytes(
    pdf_bytes: bytes,
    model: torch.nn.Module,
    tokenizer: Any,
    jobs: List[Dict],
) -> Dict[str, Any]:
    buf = io.BytesIO(pdf_bytes)
    text = preprocess_data(buf)
    entities = predict(model, tokenizer, idx2tag, rp.DEVICE, text, rp.MAX_LEN)
    merged = merge_entities_by_offsets(entities, text, verbose=False)
    skills_raw = skills_for_normalizer(merged)
    normalized, _audit = rp.SKILL_NORMALIZER.normalize(skills_raw)
    skills_canonical = [n["canonical"] for n in normalized]
    matched = rank_jobs_by_skill_overlap(skills_canonical, jobs, top_k=20)
    jobs_out = [
        {
            "company": m.get("company_norm"),
            "title": m.get("title_clean"),
            "location": m.get("location_clean"),
            "score": m.get("score"),
            "matched_skills": m.get("matched_skills") or [],
        }
        for m in matched
    ]
    return {"skills": skills_canonical, "jobs": jobs_out}


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs = rp._load_jobs_or_none(rp.JOB_POSTINGS_PATH)
    app.state.jobs = jobs if jobs is not None else []
    loaded = rp._load_ner_model_or_none()
    if loaded is None:
        app.state.model = None
        app.state.tokenizer = None
    else:
        app.state.model, app.state.tokenizer = loaded
    yield


app = FastAPI(title="简历解析", lifespan=lifespan)

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>简历技能与岗位匹配</title>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2332;
      --border: #2d3a4f;
      --text: #e8eef5;
      --muted: #8fa3b8;
      --accent: #5b9fd4;
      --accent-dim: #3d7fb0;
      --tag: #243447;
      --danger: #c45c5c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "SF Pro Text", "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(1200px 600px at 20% -10%, #1e3a5f 0%, transparent 55%),
                  radial-gradient(800px 400px at 90% 0%, #2a1f4a 0%, transparent 50%),
                  var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .wrap { max-width: 880px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
    h1 { font-size: 1.65rem; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 0.35rem; }
    .sub { color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    label.upload {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 0.75rem;
      min-height: 140px;
      border: 2px dashed var(--border);
      border-radius: 10px;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }
    label.upload:hover, label.upload.drag { border-color: var(--accent); background: rgba(91,159,212,0.06); }
    label.upload input { display: none; }
    .hint { color: var(--muted); font-size: 0.9rem; }
    button.run {
      margin-top: 1rem;
      width: 100%;
      padding: 0.75rem 1rem;
      font-size: 1rem;
      font-weight: 600;
      color: #fff;
      background: linear-gradient(180deg, var(--accent), var(--accent-dim));
      border: none;
      border-radius: 8px;
      cursor: pointer;
    }
    button.run:disabled { opacity: 0.45; cursor: not-allowed; }
    button.run:not(:disabled):hover { filter: brightness(1.06); }
    .section-title { font-size: 1.05rem; font-weight: 600; margin: 0 0 0.75rem; }
    .tags { display: flex; flex-wrap: wrap; gap: 0.45rem; }
    .tag {
      background: var(--tag);
      border: 1px solid var(--border);
      padding: 0.25rem 0.55rem;
      border-radius: 6px;
      font-size: 0.85rem;
    }
    .job {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      margin-bottom: 0.65rem;
    }
    .job:last-child { margin-bottom: 0; }
    .job-head { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: baseline; }
    .job-title { font-weight: 600; }
    .score { color: var(--accent); font-size: 0.85rem; }
    .meta { color: var(--muted); font-size: 0.88rem; margin-top: 0.25rem; }
    .err { color: #ffb4b4; background: rgba(196,92,92,0.15); border: 1px solid var(--danger); padding: 0.75rem 1rem; border-radius: 8px; margin-top: 1rem; display: none; }
    .empty { color: var(--muted); font-size: 0.9rem; }
    #status { color: var(--muted); font-size: 0.88rem; margin-top: 0.5rem; min-height: 1.25em; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>简历技能与岗位匹配</h1>
    <p class="sub">上传 PDF 简历，自动提取<strong>所会技能</strong>并推荐<strong>适合岗位</strong>。</p>

    <div class="card">
      <label class="upload" id="drop">
        <span id="fname" class="hint">点击或拖拽 PDF 到此处</span>
        <input type="file" id="file" accept=".pdf,application/pdf" />
      </label>
      <button type="button" class="run" id="go" disabled>分析简历</button>
      <div id="status"></div>
      <div class="err" id="err"></div>
    </div>

    <div class="card" id="skillsCard" style="display:none">
      <h2 class="section-title">所会技能</h2>
      <div class="tags" id="skills"></div>
    </div>

    <div class="card" id="jobsCard" style="display:none">
      <h2 class="section-title">适合岗位</h2>
      <div id="jobs"></div>
    </div>
  </div>
  <script>
    const drop = document.getElementById("drop");
    const input = document.getElementById("file");
    const go = document.getElementById("go");
    const fname = document.getElementById("fname");
    const status = document.getElementById("status");
    const err = document.getElementById("err");
    const skillsCard = document.getElementById("skillsCard");
    const jobsCard = document.getElementById("jobsCard");
    const skillsEl = document.getElementById("skills");
    const jobsEl = document.getElementById("jobs");
    let file = null;

    function setFile(f) {
      file = f;
      go.disabled = !f;
      fname.textContent = f ? f.name : "点击或拖拽 PDF 到此处";
      err.style.display = "none";
    }
    input.addEventListener("change", () => {
      const f = input.files && input.files[0];
      if (f) setFile(f);
    });
    ["dragenter","dragover"].forEach((ev) => {
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); });
    });
    ["dragleave","drop"].forEach((ev) => {
      drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); });
    });
    drop.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f && f.name.toLowerCase().endsWith(".pdf")) setFile(f);
    });

    go.addEventListener("click", async () => {
      if (!file) return;
      err.style.display = "none";
      skillsCard.style.display = "none";
      jobsCard.style.display = "none";
      go.disabled = true;
      status.textContent = "正在分析…";
      const fd = new FormData();
      fd.append("file", file);
      try {
        const r = await fetch("/api/analyze", { method: "POST", body: fd });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          throw new Error(data.detail || r.statusText || "请求失败");
        }
        const skills = data.skills || [];
        const jobs = data.jobs || [];
        skillsEl.innerHTML = skills.length
          ? skills.map((s) => `<span class="tag">${escapeHtml(s)}</span>`).join("")
          : '<span class="empty">未识别到技能（或词典未覆盖）。</span>';
        skillsCard.style.display = "block";
        if (jobs.length) {
          jobsEl.innerHTML = jobs.map(renderJob).join("");
        } else {
          jobsEl.innerHTML = '<p class="empty">暂无匹配岗位（职位库为空或没有技能重叠）。</p>';
        }
        jobsCard.style.display = "block";
        status.textContent = "完成";
      } catch (e) {
        err.textContent = e.message || String(e);
        err.style.display = "block";
        status.textContent = "";
      } finally {
        go.disabled = !file;
      }
    });

    function escapeHtml(s) {
      return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
    }
    function renderJob(j) {
      const loc = j.location ? ` · ${escapeHtml(j.location)}` : "";
      const co = j.company ? escapeHtml(j.company) : "—";
      const ti = j.title ? escapeHtml(j.title) : "—";
      const ms = (j.matched_skills || []).slice(0, 8).map(escapeHtml).join("、");
      const more = (j.matched_skills || []).length > 8 ? "…" : "";
      return `<div class="job"><div class="job-head"><span class="job-title">${ti}</span><span class="score">匹配技能数：${j.score ?? 0}</span></div><div class="meta">${co}${loc}</div>${ms ? `<div class="meta" style="margin-top:0.4rem">重叠技能：${ms}${more}</div>` : ""}</div>`;
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "model_loaded": app.state.model is not None,
        "job_count": len(getattr(app.state, "jobs", []) or []),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 简历")
    if app.state.model is None or app.state.tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="NER 模型未加载，请检查 model-state.bin 与词表路径",
        )
    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="文件为空")
    try:
        return _analyze_pdf_bytes(body, app.state.model, app.state.tokenizer, app.state.jobs)
    except Exception as exc:  # noqa: BLE001 — surface pipeline errors to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc
