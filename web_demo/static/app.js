"use strict";
// The visual intro does not delay model initialization or page interaction.
const introScreen = document.getElementById("intro-screen");
introScreen.addEventListener("animationend", event => {
  if (event.animationName === "intro-exit") introScreen.remove();
});
const $ = id => document.getElementById(id);
const state = { file: null, bitmap: null, ready: false, busy: false, result: null, resultView: false, downloadUrl: null, choice: 0 };
const palette = ["#18633b", "#a4471d", "#315ea1", "#803d86", "#7b6510"];
let healthTimer;

function message(text) { $("error").textContent = text; $("error").hidden = !text; }
function controls() {
  $("detect").disabled = !state.ready || !state.file || state.busy;
  $("file-input").disabled = state.busy;
  $("replace").disabled = state.busy;
  $("reset").disabled = state.busy;
  $("download").disabled = state.busy || !state.downloadUrl;
  $("processing-overlay").hidden = !state.busy;
  $("detect-label").textContent = state.busy ? "Processing image…" : "Detect crop condition";
}
function releaseDownload() { if (state.downloadUrl) URL.revokeObjectURL(state.downloadUrl); state.downloadUrl = null; }
function emptyResults() {
  state.result = null; state.resultView = false; releaseDownload();
  $("results-empty").hidden = false; $("result-content").hidden = true;
  $("result-state").textContent = state.file ? "Ready to detect" : "Awaiting image";
  $("result-tab").disabled = true;
  $("detections").replaceChildren();
  selectView(false);
}
function selectView(resultView) {
  state.resultView = resultView;
  $("original-tab").classList.toggle("selected", !resultView);
  $("result-tab").classList.toggle("selected", resultView);
  $("original-tab").setAttribute("aria-pressed", String(!resultView));
  $("result-tab").setAttribute("aria-pressed", String(resultView));
  draw(resultView);
}
function draw(withBoxes) {
  if (!state.bitmap) return;
  const canvas = $("image-canvas"), ctx = canvas.getContext("2d");
  canvas.width = state.bitmap.width; canvas.height = state.bitmap.height;
  ctx.drawImage(state.bitmap, 0, 0);
  if (!withBoxes || !state.result) return;
  const fontSize = Math.max(14, Math.round(Math.max(canvas.width, canvas.height) / 65));
  const lineWidth = Math.max(2, fontSize / 6);
  ctx.font = `600 ${fontSize}px "Segoe UI", sans-serif`;
  state.result.detections.forEach((d, i) => {
    const [x1,y1,x2,y2] = d.box, color = palette[i % palette.length];
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    let label = `${d.crop}: ${d.condition} ${(d.confidence*100).toFixed(1)}%`;
    while (ctx.measureText(label).width > canvas.width-12 && label.length > 10) label = label.slice(0,-5) + "…";
    const w = Math.min(canvas.width, ctx.measureText(label).width + 12), h = fontSize+12;
    const lx = Math.min(Math.max(0,x1), Math.max(0,canvas.width-w));
    const ly = Math.max(0, Math.min(canvas.height-h, y1 >= h ? y1-h : y1));
    ctx.fillStyle = color; ctx.fillRect(lx,ly,w,h);
    ctx.fillStyle = "#fff"; ctx.fillText(label,lx+6,ly+fontSize+2);
  });
}
async function choose(file) {
  if (!file || state.busy) return;
  const choice = ++state.choice;
  message("");
  const type = file.type || ({jpg:"image/jpeg",jpeg:"image/jpeg",png:"image/png",webp:"image/webp"}[file.name.split(".").pop().toLowerCase()]);
  if (!["image/jpeg","image/png","image/webp"].includes(type)) { message("Choose a JPG, PNG or WebP image."); return; }
  if (file.size > 10*1024*1024) { message("Choose an image smaller than 10 MB."); return; }
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, {imageOrientation:"from-image"});
    if (bitmap.width*bitmap.height > 20000000) { bitmap.close(); message("Use an image under 20 megapixels."); return; }
  } catch { message("This image could not be read. Try another photograph."); return; }
  if (choice !== state.choice) { bitmap.close(); return; }
  if (state.bitmap) state.bitmap.close();
  state.bitmap = bitmap; state.file = file; state.contentType = type;
  emptyResults();
  $("dropzone").hidden = true; $("preview").hidden = false; $("reset").hidden = false;
  $("filename").textContent = file.name;
  $("dimensions").textContent = `${bitmap.width} × ${bitmap.height} px`;
  selectView(false); controls();
}
function reset() {
  if (state.busy) return;
  ++state.choice;
  state.file = null;
  if (state.bitmap) state.bitmap.close(); state.bitmap = null;
  $("file-input").value = "";
  $("image-canvas").width = 1; $("image-canvas").height = 1;
  $("preview").hidden = true; $("dropzone").hidden = false; $("reset").hidden = true;
  emptyResults(); message(""); controls();
}
function renderResult(result) {
  state.result = result;
  $("results-empty").hidden = true; $("result-content").hidden = false;
  const count = result.detections.length;
  $("result-state").textContent = "Complete";
  $("result-title").textContent = count ? `${count} ${count===1?"region":"regions"} detected` : "No reliable detection";
  $("result-message").textContent = count ? "Review the marked regions and predicted conditions below." : "Try a clearer image of a supported crop. The condition may be outside the model’s 24 classes. No detection does not establish that the plant is healthy.";
  const list = $("detections"); list.replaceChildren();
  result.detections.forEach((d, i) => {
    const row = document.createElement("div"); row.className = "detection";
    const crop = document.createElement("span"); crop.className = "crop"; crop.textContent = d.crop;
    const condition = document.createElement("span"); condition.className = "condition"; condition.textContent = d.condition;
    const score = document.createElement("span"); score.className = "score"; score.textContent = `${(d.confidence*100).toFixed(1)}%`;
    const track = document.createElement("div"), fill = document.createElement("div"); track.className="score-track"; fill.className="score-fill";
    fill.style.width = `${Math.max(0, Math.min(100,d.confidence*100))}%`; fill.style.backgroundColor=palette[i%palette.length]; track.append(fill);
    row.append(crop,condition,score,track); list.append(row);
  });
  $("timing").textContent = `Model inference: ${(result.inference_ms/1000).toFixed(2)} s · Request processing: ${(result.processing_ms/1000).toFixed(2)} s`;
  $("result-tab").disabled = false; selectView(true);
  releaseDownload();
  if (count) {
    const expected = state.result;
    $("image-canvas").toBlob(blob => {
      if (blob && state.result===expected) { state.downloadUrl = URL.createObjectURL(blob); controls(); }
    }, "image/png");
  }
}
async function detect() {
  if (!state.file || !state.ready || state.busy) return;
  emptyResults(); state.busy = true; message(""); controls();
  $("result-state").textContent = "Processing";
  const abort = new AbortController(), timer = setTimeout(()=>abort.abort(),120000);
  try {
    const response = await fetch("/api/predict", {method:"POST",headers:{"Content-Type":state.contentType},body:state.file,signal:abort.signal});
    let result;
    try { result = await response.json(); }
    catch { throw new Error("The service returned an unreadable response. Try again in a moment."); }
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "The image could not be processed. Try again.");
    if (result.width !== state.bitmap.width || result.height !== state.bitmap.height) throw new Error("The image orientation could not be matched. Export the photograph as PNG and try again.");
    renderResult(result);
    if (window.matchMedia("(max-width:740px)").matches) $("results-heading").scrollIntoView({block:"start",behavior:"auto"});
  } catch (error) {
    message(error.name === "AbortError" ? "Processing took longer than expected. Wait a moment before retrying." : error.message === "Failed to fetch" ? "The local application could not be reached. Check that it is running." : error.message);
    $("result-state").textContent = "Try again";
  } finally { clearTimeout(timer); state.busy = false; controls(); }
}
async function health() {
  const abort = new AbortController(), timeout = setTimeout(()=>abort.abort(),5000);
  try {
    const r = await fetch("/api/health", {signal:abort.signal}); if (!r.ok) throw new Error(); const info=await r.json();
    state.ready=info.ready;
    $("connection").className = `connection ${info.state}`;
    $("connection-text").textContent = info.ready ? "Local model ready" : info.state === "unavailable" ? "Model unavailable" : "Loading model";
    $("model-info").textContent = `${info.model} · ${info.device.toUpperCase()}${info.provisional ? ". Provisional model while tuned-model selection is in progress." : "."}`;
    $("threshold-note").textContent = info.threshold_validated ? `Display threshold: ${(info.threshold*100).toFixed(0)}%.` : `Demo display threshold: ${(info.threshold*100).toFixed(0)}%; validation calibration is pending. This does not reliably reject unfamiliar diseases.`;
    if (info.classes.length && !$("class-list").childElementCount) {
      info.supported_crops.forEach(crop => {
        const section=document.createElement("div"), title=document.createElement("strong"), list=document.createElement("ul");
        section.className="class-group"; title.textContent=crop;
        info.classes.filter(c=>c.crop===crop).forEach(c=>{const li=document.createElement("li");li.textContent=c.condition;list.append(li);});
        section.append(title,list);$("class-list").append(section);
      });
    }
  } catch { state.ready=false; $("connection").className="connection unavailable"; $("connection-text").textContent="Service unreachable"; }
  finally { clearTimeout(timeout); }
  controls(); healthTimer=setTimeout(health,state.ready?20000:2500);
}
$("file-input").addEventListener("change",e=>choose(e.target.files[0]));
$("replace").addEventListener("click",()=>{ $("file-input").value=""; $("file-input").click(); });
$("reset").addEventListener("click",reset); $("detect").addEventListener("click",detect);
$("original-tab").addEventListener("click",()=>selectView(false)); $("result-tab").addEventListener("click",()=>selectView(true));
$("download").addEventListener("click",()=>{if(!state.downloadUrl)return;const a=document.createElement("a");a.href=state.downloadUrl;a.download="AgriVision-detections.png";a.click();});
$("dropzone").addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();if(!state.busy)$("file-input").click();}});
for (const name of ["dragenter","dragover"]) $("dropzone").addEventListener(name,e=>{e.preventDefault();if(!state.busy)$("dropzone").classList.add("dragging");});
for (const name of ["dragleave","drop"]) $("dropzone").addEventListener(name,e=>{e.preventDefault();$("dropzone").classList.remove("dragging");});
$("dropzone").addEventListener("drop",e=>choose(e.dataTransfer.files[0]));
window.addEventListener("beforeunload",()=>{releaseDownload();clearTimeout(healthTimer);if(state.bitmap)state.bitmap.close();});
health();
