// Browser port of estimating-personality-from-a-facial-photo.py
// Pipeline (mirrors DeepFace.represent with Facenet512): detect face -> level the eyes -> crop ->
// letterbox to 160x160 -> scale to [0, 1] -> FaceNet-512 -> L2-normalise -> cosine similarity.

import { FaceDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
const MP_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'models/facenet512.onnx';
const DETECTOR_URL = 'models/blaze_face_short_range.tflite';
const INPUT_SIZE = 160;
const MAX_SIDE = 1280; // photos are downscaled to this before detection

// The crop is a square placed from the two eye positions. These proportions were measured from the
// boxes that DeepFace's default OpenCV detector (used by the Python script) puts on the same faces.
const CROP = { sidePerEyeDistance: 2.7, eyeLineFromTop: 0.39 };

// "A" composites average people high in the trait, "B" composites people low in it.
// Neuroticism is reported reversed, as emotional stability, exactly like the original script.
const TRAITS = [
  { key: 'Agr', name: 'Agreeableness', hi: 'More agreeable', lo: 'Less agreeable', mid: 'Moderately agreeable', reversed: false },
  { key: 'Con', name: 'Conscientiousness', hi: 'More conscientious', lo: 'Less conscientious', mid: 'Moderately conscientious', reversed: false },
  { key: 'Ext', name: 'Extraversion', hi: 'More extraverted', lo: 'Less extraverted', mid: 'Moderately extraverted', reversed: false },
  { key: 'Neu', name: 'Emotional stability', hi: 'More emotionally stable', lo: 'Less emotionally stable', mid: 'Moderately emotionally stable', reversed: true },
  { key: 'Ope', name: 'Openness', hi: 'More open', lo: 'Less open', mid: 'Moderately open', reversed: false },
];

const $ = (id) => document.getElementById(id);
const els = {
  tabUpload: $('tab-upload'), tabCamera: $('tab-camera'),
  panelUpload: $('panel-upload'), panelCamera: $('panel-camera'),
  dropzone: $('dropzone'), fileInput: $('file-input'),
  video: $('video'), camMsg: $('cam-msg'), camStart: $('cam-start'), camShoot: $('cam-shoot'), camStop: $('cam-stop'),
  photoCanvas: $('photo-canvas'), printEmpty: $('print-empty'), printCaption: $('print-caption'),
  run: $('run'), runNote: $('run-note'),
  progress: $('progress'), progressFill: $('progress-fill'), progressLabel: $('progress-label'),
  error: $('error'), results: $('results'), traits: $('traits'), axisNote: $('axis-note'), cropCanvas: $('crop-canvas'),
};

const state = { hasPhoto: false, gender: null, busy: false, stream: null };
let sessionPromise = null;
let detectorPromise = null;
const compositeCache = {}; // gender -> { Agr_MA: Float32Array, ... }

// ---------- UI: tabs ----------
function selectTab(which) {
  const upload = which === 'upload';
  els.tabUpload.setAttribute('aria-selected', String(upload));
  els.tabCamera.setAttribute('aria-selected', String(!upload));
  els.tabUpload.tabIndex = upload ? 0 : -1;
  els.tabCamera.tabIndex = upload ? -1 : 0;
  els.panelUpload.hidden = !upload;
  els.panelCamera.hidden = upload;
  if (upload) stopCamera();
}
els.tabUpload.addEventListener('click', () => selectTab('upload'));
els.tabCamera.addEventListener('click', () => selectTab('camera'));
for (const tab of [els.tabUpload, els.tabCamera]) {
  tab.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const other = tab === els.tabUpload ? els.tabCamera : els.tabUpload;
    other.focus();
    other.click();
  });
}

// ---------- UI: photo intake ----------
function setPhoto(source, width, height, caption) {
  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  const c = els.photoCanvas;
  c.width = Math.round(width * scale);
  c.height = Math.round(height * scale);
  c.getContext('2d').drawImage(source, 0, 0, c.width, c.height);
  c.hidden = false;
  els.printEmpty.hidden = true;
  els.printCaption.textContent = caption;
  state.hasPhoto = true;
  els.results.hidden = true;
  showError(null);
  refreshRunButton();
}

async function loadFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showError('That file is not an image. Please choose a JPG, PNG or WebP photo.');
    return;
  }
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    setPhoto(bitmap, bitmap.width, bitmap.height, `Uploaded: ${file.name}`);
    bitmap.close();
  } catch {
    showError('This image could not be opened. Please try a different file.');
  }
}

els.fileInput.addEventListener('change', () => loadFile(els.fileInput.files[0]));
for (const type of ['dragenter', 'dragover']) {
  els.dropzone.addEventListener(type, (e) => { e.preventDefault(); els.dropzone.classList.add('is-over'); });
}
for (const type of ['dragleave', 'drop']) {
  els.dropzone.addEventListener(type, (e) => { e.preventDefault(); els.dropzone.classList.remove('is-over'); });
}
els.dropzone.addEventListener('drop', (e) => loadFile(e.dataTransfer.files[0]));

// ---------- UI: camera ----------
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    els.camMsg.textContent = 'This browser does not give web pages access to a camera.';
    return;
  }
  els.camMsg.textContent = 'Waiting for permission to use the camera…';
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
    });
    els.video.srcObject = state.stream;
    await els.video.play();
    els.camMsg.hidden = true;
    els.camStart.hidden = true;
    els.camShoot.hidden = false;
    els.camStop.hidden = false;
  } catch (err) {
    const denied = err?.name === 'NotAllowedError' || err?.name === 'SecurityError';
    const missing = err?.name === 'NotFoundError' || err?.name === 'OverconstrainedError';
    els.camMsg.hidden = false;
    els.camMsg.textContent = denied
      ? 'Camera access was blocked. Allow it in the browser’s site settings, then try again.'
      : missing ? 'No camera was found on this device.'
      : 'The camera could not be started. It may be in use by another app.';
  }
}

function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  els.video.srcObject = null;
  els.camMsg.hidden = false;
  els.camMsg.textContent = 'The camera is off.';
  els.camStart.hidden = false;
  els.camShoot.hidden = true;
  els.camStop.hidden = true;
}

els.camStart.addEventListener('click', startCamera);
els.camStop.addEventListener('click', stopCamera);
els.camShoot.addEventListener('click', () => {
  const v = els.video;
  if (!v.videoWidth) return;
  // The preview is mirrored for comfort; the captured photo is not, so it matches a normal photograph.
  setPhoto(v, v.videoWidth, v.videoHeight, `Camera capture, ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`);
});

// ---------- UI: gender + run ----------
document.querySelectorAll('input[name="gender"]').forEach((input) => {
  input.addEventListener('change', () => { state.gender = input.value; refreshRunButton(); });
});

function refreshRunButton() {
  const ready = state.hasPhoto && state.gender && !state.busy;
  els.run.disabled = !ready;
  els.runNote.textContent = state.busy ? ''
    : !state.hasPhoto && !state.gender ? 'Add a photo and pick a composite set first.'
    : !state.hasPhoto ? 'Add a photo first.'
    : !state.gender ? 'Pick a composite set first.'
    : 'Ready. The first run downloads the model (about 45 MB).';
}

function showError(message) {
  els.error.hidden = !message;
  els.error.textContent = message || '';
}

function setProgress(fraction, label) {
  els.progress.hidden = false;
  els.progressFill.style.transform = `scaleX(${Math.max(0, Math.min(1, fraction))})`;
  els.progressLabel.textContent = label;
}

// Yields to the browser so the progress bar can repaint (a timer, so it also works in a background tab).
const nextFrame = () => new Promise((r) => setTimeout(r, 16));

// ---------- Models ----------
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}) for ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(1, received / total), received);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out;
}

function getSession(onProgress) {
  sessionPromise ??= (async () => {
    ort.env.wasm.wasmPaths = ORT_DIST;
    const bytes = await fetchWithProgress(MODEL_URL, onProgress);
    return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  })().catch((err) => { sessionPromise = null; throw err; });
  return sessionPromise;
}

function getDetector() {
  detectorPromise ??= (async () => {
    const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
    return FaceDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: DETECTOR_URL, delegate: 'CPU' },
      runningMode: 'IMAGE',
      minDetectionConfidence: 0.5,
    });
  })().catch((err) => { detectorPromise = null; throw err; });
  return detectorPromise;
}

// ---------- Face detection, alignment, crop ----------
function largestFace(detections) {
  let best = null;
  for (const d of detections) {
    const b = d.boundingBox;
    if (!b || !d.keypoints || d.keypoints.length < 2) continue;
    if (!best || b.width * b.height > best.boundingBox.width * best.boundingBox.height) best = d;
  }
  return best;
}

// Returns { box: {x, y, w, h}, angle } in the coordinates of `canvas`, or null when no face is found.
// Tightly cropped faces (like the composites) are retried with a margin, which the detector needs.
function findFace(detector, canvas) {
  for (const pad of [0, 0.5]) {
    let target = canvas;
    let ox = 0, oy = 0;
    if (pad) {
      ox = Math.round(canvas.width * pad);
      oy = Math.round(canvas.height * pad);
      target = document.createElement('canvas');
      target.width = canvas.width + 2 * ox;
      target.height = canvas.height + 2 * oy;
      const ctx = target.getContext('2d');
      ctx.fillStyle = '#808080';
      ctx.fillRect(0, 0, target.width, target.height);
      ctx.drawImage(canvas, ox, oy);
    }
    const face = largestFace(detector.detect(target).detections);
    if (!face) continue;
    const [eyeA, eyeB] = face.keypoints; // subject's right eye, then left eye (normalised coordinates)
    const ax = eyeA.x * target.width - ox, ay = eyeA.y * target.height - oy;
    const bx = eyeB.x * target.width - ox, by = eyeB.y * target.height - oy;
    const side = Math.hypot(bx - ax, by - ay) * CROP.sidePerEyeDistance;
    const cx = (ax + bx) / 2, cy = (ay + by) / 2;
    return {
      box: { x: cx - side / 2, y: cy - side * CROP.eyeLineFromTop, w: side, h: side },
      pivot: { x: cx, y: cy },
      angle: Math.atan2(by - ay, bx - ax),
    };
  }
  return null;
}

// Rotates the image about the face centre so the eyes are level, crops the box, and letterboxes it
// into a 160x160 black square (DeepFace's resize-with-padding).
function cropFace(canvas, face) {
  const out = document.createElement('canvas');
  out.width = INPUT_SIZE;
  out.height = INPUT_SIZE;
  const ctx = out.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  const { x, y, w, h } = face.box;
  const scale = INPUT_SIZE / Math.max(w, h);
  ctx.save();
  ctx.beginPath();
  ctx.rect((INPUT_SIZE - w * scale) / 2, (INPUT_SIZE - h * scale) / 2, w * scale, h * scale);
  ctx.clip();
  ctx.imageSmoothingQuality = 'high';
  const pivot = face.pivot ?? { x: x + w / 2, y: y + h / 2 };
  ctx.translate((INPUT_SIZE - w * scale) / 2, (INPUT_SIZE - h * scale) / 2);
  ctx.scale(scale, scale);
  ctx.translate(pivot.x - x, pivot.y - y);
  ctx.rotate(-face.angle);
  ctx.translate(-pivot.x, -pivot.y);
  ctx.drawImage(canvas, 0, 0);
  ctx.restore();
  return out;
}

// ---------- Embedding ----------
async function embed(session, faceCanvas) {
  const { data } = faceCanvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const input = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    // Channel order is blue, green, red: that is what DeepFace.represent feeds this model.
    input[j] = data[i + 2] / 255;
    input[j + 1] = data[i + 1] / 255;
    input[j + 2] = data[i] / 255;
  }
  const tensor = new ort.Tensor('float32', input, [1, INPUT_SIZE, INPUT_SIZE, 3]);
  const output = await session.run({ [session.inputNames[0]]: tensor });
  const vec = Float32Array.from(output[session.outputNames[0]].data);
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  return vec.map((v) => v / norm);
}

const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load ${src}`));
    img.src = src;
  });
}

async function embedComposite(session, detector, name) {
  const img = await loadImage(`images/${name}.png`);
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext('2d').drawImage(img, 0, 0);
  const face = findFace(detector, canvas)
    ?? { box: { x: 0, y: 0, w: canvas.width, h: canvas.height }, angle: 0 };
  return embed(session, cropFace(canvas, face));
}

// ---------- Run ----------
async function run() {
  if (state.busy || !state.hasPhoto || !state.gender) return;
  state.busy = true;
  refreshRunButton();
  showError(null);
  els.results.hidden = true;
  const gender = state.gender;

  try {
    setProgress(0.02, 'Loading the face-recognition model…');
    const [session, detector] = await Promise.all([
      getSession((p, bytes) => setProgress(0.02 + 0.5 * p, `Downloading the model… ${(bytes / 1048576).toFixed(0)} MB`)),
      getDetector(),
    ]);

    setProgress(0.55, 'Looking for a face in your photo…');
    await nextFrame();
    const face = findFace(detector, els.photoCanvas);
    if (!face) {
      throw new UserError('No face was found in this photo. Try one where a single face is clearly visible, facing the camera, and well lit.');
    }
    const crop = cropFace(els.photoCanvas, face);
    const photoVec = await embed(session, crop);

    const names = TRAITS.flatMap((t) => [`${t.key}_${gender}A`, `${t.key}_${gender}B`]);
    compositeCache[gender] ??= {};
    const cache = compositeCache[gender];
    let done = 0;
    for (const name of names) {
      setProgress(0.6 + 0.4 * (done / names.length), `Comparing with composite faces… ${done + 1} of ${names.length}`);
      if (!cache[name]) {
        await nextFrame();
        cache[name] = await embedComposite(session, detector, name);
      }
      done++;
    }

    const rows = TRAITS.map((t) => {
      const simA = cosine(photoVec, cache[`${t.key}_${gender}A`]);
      const simB = cosine(photoVec, cache[`${t.key}_${gender}B`]);
      return {
        trait: t,
        simHi: t.reversed ? simB : simA,
        simLo: t.reversed ? simA : simB,
        imgHi: `images/${t.key}_${gender}${t.reversed ? 'B' : 'A'}.png`,
        imgLo: `images/${t.key}_${gender}${t.reversed ? 'A' : 'B'}.png`,
      };
    });

    setProgress(1, 'Done.');
    renderResults(rows, crop, gender);
    els.progress.hidden = true;
  } catch (err) {
    els.progress.hidden = true;
    console.error(err);
    showError(err instanceof UserError ? err.message
      : 'Something went wrong while loading the models or analysing the photo. Check your connection and try again.');
  } finally {
    state.busy = false;
    refreshRunButton();
  }
}

class UserError extends Error {}

els.run.addEventListener('click', run);

// ---------- Results ----------
const fmt = (v) => v.toFixed(3);
const signed = (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(3);

function renderResults(rows, crop, gender) {
  els.cropCanvas.getContext('2d').drawImage(crop, 0, 0);

  const maxGap = Math.max(...rows.map((r) => Math.abs(r.simHi - r.simLo)));
  const axisMax = Math.max(0.05, Math.ceil(maxGap * 20) / 20);
  const setName = gender === 'M' ? 'male' : 'female';

  els.traits.classList.remove('is-drawn');
  els.traits.replaceChildren(...rows.map((r) => {
    const gap = r.simHi - r.simLo;
    const verdict = gap > 0 ? r.trait.hi : gap < 0 ? r.trait.lo : r.trait.mid;
    const pct = (Math.abs(gap) / axisMax) * 50;
    const row = document.createElement('article');
    row.className = 'trait';
    row.innerHTML = `
      <div class="trait-head">
        <span class="trait-name">${r.trait.name}</span>
        <span class="trait-verdict">${verdict}</span>
        <span class="trait-diff">Abs. diff. in cosine sim.: ${fmt(Math.abs(gap))}</span>
      </div>
      <div class="trait-body">
        <figure class="pole pole-lo ${gap < 0 ? 'is-winner' : ''}">
          <img src="${r.imgLo}" alt="Composite ${setName} face: ${r.trait.lo.toLowerCase()}" width="84" height="84">
          <figcaption><span class="pole-label">${r.trait.lo}</span><span class="pole-sim">cos ${fmt(r.simLo)}</span></figcaption>
        </figure>
        <div class="gap" role="img" aria-label="Gap in cosine similarity: ${signed(gap)}, toward ${verdict.toLowerCase()}">
          <div class="gap-track"></div>
          <div class="gap-bar ${gap >= 0 ? 'to-hi' : 'to-lo'}" style="width:${pct}%"></div>
          <div class="gap-zero"></div>
          <span class="gap-value ${gap >= 0 ? 'to-hi' : 'to-lo'}" style="left:${gap >= 0 ? 50 + pct : 50 - pct}%">${signed(gap)}</span>
        </div>
        <figure class="pole pole-hi ${gap > 0 ? 'is-winner' : ''}">
          <img src="${r.imgHi}" alt="Composite ${setName} face: ${r.trait.hi.toLowerCase()}" width="84" height="84">
          <figcaption><span class="pole-label">${r.trait.hi}</span><span class="pole-sim">cos ${fmt(r.simHi)}</span></figcaption>
        </figure>
      </div>`;
    return row;
  }));

  els.axisNote.textContent = `Bar scale: the full width of a bar track runs from −${axisMax.toFixed(2)} to +${axisMax.toFixed(2)} in cosine-similarity gap, the same for every trait. Cosine similarity itself ranges from −1 to 1.`;
  els.results.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => els.traits.classList.add('is-drawn')));
  els.results.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
}

// Exposed for verification against the Python script.
window.__faceApp = { getSession, getDetector, findFace, cropFace, embed, embedComposite, cosine };
