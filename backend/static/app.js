'use strict';

// ── State ──────────────────────────────────────────────────
let selectedFile = null;
let analysis     = null;
let recommend    = null;
let cameraStream = null;
let facingMode   = 'user';       // 'user' = front, 'environment' = back

// ── DOM ────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const methodGallery  = $('methodGallery');
const methodCamera   = $('methodCamera');
const fileInput      = $('fileInput');
const previewWrap    = $('previewWrap');
const previewImg     = $('previewImg');
const changeBtn      = $('changeBtn');
const analyzeBtn     = $('analyzeBtn');
const uploadSection  = $('uploadSection');
const loadingSection = $('loadingSection');
const loadingText    = $('loadingText');
const resultsSection = $('resultsSection');

// Camera
const cameraModal  = $('cameraModal');
const cameraVideo  = $('cameraVideo');
const cameraCanvas = $('cameraCanvas');
const captureBtn   = $('captureBtn');
const cameraClose  = $('cameraClose');
const flipBtn      = $('flipBtn');

// Results
const profileItems  = $('profileItems');
const swatchesEl    = $('swatches');
const outfitTags    = $('outfitTags');
const accessoryTags = $('accessoryTags');
const styleDesc     = $('styleDesc');
const tipsList      = $('tipsList');
const genStatus     = $('genStatus');
const genImages     = $('genImages');
const tryAgainBtn   = $('tryAgainBtn');

// ── Color CSS map ───────────────────────────────────────────
const COLOR_CSS = {
  'ivory': '#fffff0', 'soft pink': '#ffb6c1', 'lavender': '#e6e6fa',
  'powder blue': '#b0e0e6', 'mint green': '#98ff98', 'blush rose': '#f4a7b9',
  'champagne': '#f7e7ce', 'light grey': '#d3d3d3', 'peach': '#ffcba4',
  'sky blue': '#87ceeb', 'olive': '#6b7c3a', 'terracotta': '#c36a2d',
  'forest green': '#228b22', 'rust': '#b7410e', 'coral': '#ff7f50',
  'caramel': '#c68642', 'burgundy': '#800020', 'deep teal': '#008080',
  'cinnamon': '#d2691e', 'amber': '#ffbf00', 'mustard': '#e3a857',
  'royal blue': '#4169e1', 'emerald green': '#50c878', 'gold': '#ffd700',
  'white': '#f5f5f5', 'cobalt': '#0047ab', 'fuchsia': '#ff00ff',
  'deep orange': '#ff6600', 'wine red': '#722f37', 'cream': '#fffdd0',
  'navy': '#001f5b', 'beige': '#f5f0e8',
};

// ── Upload / Gallery ────────────────────────────────────────
methodGallery.addEventListener('click', () => fileInput.click());
methodGallery.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

// Drag-and-drop on the whole upload section
uploadSection.addEventListener('dragover', e => { e.preventDefault(); methodGallery.classList.add('dragover'); });
uploadSection.addEventListener('dragleave', () => methodGallery.classList.remove('dragover'));
uploadSection.addEventListener('drop', e => {
  e.preventDefault();
  methodGallery.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) setFile(f);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

changeBtn.addEventListener('click', resetUpload);

tryAgainBtn.addEventListener('click', () => {
  resultsSection.hidden = true;
  uploadSection.hidden  = false;
  resetUpload();
});

async function setFile(file) {
  loadingSection.hidden = false;
  setStatus('Preparing image…');
  
  try {
    const resizedBlob = await resizeImage(file, 800, 800);
    selectedFile = new File([resizedBlob], file.name, { type: 'image/jpeg' });
    
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src           = e.target.result;
      $('methodGrid').hidden   = true;
      previewWrap.hidden       = false;
      $('step2Label').hidden   = false;
      analyzeBtn.disabled      = false;
      loadingSection.hidden    = true;
    };
    reader.readAsDataURL(selectedFile);
  } catch (err) {
    console.error('Resize failed:', err);
    selectedFile = file; // Fallback to original
    loadingSection.hidden = true;
  }
}

function resizeImage(file, maxWidth, maxHeight) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let width = img.width;
        let height = img.height;

        if (width > height) {
          if (width > maxWidth) {
            height *= maxWidth / width;
            width = maxWidth;
          }
        } else {
          if (height > maxHeight) {
            width *= maxHeight / height;
            height = maxHeight;
          }
        }

        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.85);
      };
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function resetUpload() {
  selectedFile             = null;
  fileInput.value          = '';
  $('methodGrid').hidden   = false;
  previewWrap.hidden       = true;
  $('step2Label').hidden   = true;
  analyzeBtn.disabled      = true;
}

// ── Camera ─────────────────────────────────────────────────
methodCamera.addEventListener('click', () => openCamera('user'));
methodCamera.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') openCamera('user'); });

cameraClose.addEventListener('click', closeCamera);
captureBtn.addEventListener('click', capturePhoto);
cameraModal.addEventListener('click', e => { if (e.target === cameraModal) closeCamera(); });

flipBtn.addEventListener('click', () => {
  const next = facingMode === 'user' ? 'environment' : 'user';
  openCamera(next);
});

async function openCamera(mode) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Camera is not supported in this browser.\nPlease use Chrome, Safari, or Firefox and ensure you are on HTTPS or localhost.');
    return;
  }

  // Stop any existing stream first
  stopStream();

  facingMode = mode;

  // Apply mirror only for front (selfie) camera
  cameraVideo.style.transform = mode === 'user' ? 'scaleX(-1)' : 'none';

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: mode },
        width:  { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    cameraVideo.srcObject = cameraStream;
    cameraModal.hidden = false;
    document.body.style.overflow = 'hidden';
  } catch (err) {
    const msg = err.name === 'NotAllowedError'
      ? 'Camera permission denied.\nPlease allow camera access in your browser settings and try again.'
      : err.name === 'NotFoundError'
      ? 'No camera found on this device.\nPlease upload a photo from your gallery instead.'
      : `Camera error: ${err.message}`;
    alert(msg);
  }
}

function stopStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
}

function closeCamera() {
  stopStream();
  cameraVideo.srcObject = null;
  cameraModal.hidden = true;
  document.body.style.overflow = '';
}

function capturePhoto() {
  const vw = cameraVideo.videoWidth  || 1280;
  const vh = cameraVideo.videoHeight || 720;
  cameraCanvas.width  = vw;
  cameraCanvas.height = vh;

  const ctx = cameraCanvas.getContext('2d');

  if (facingMode === 'user') {
    // Mirror canvas so captured image matches what user saw
    ctx.translate(vw, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(cameraVideo, 0, 0, vw, vh);

  cameraCanvas.toBlob(blob => {
    if (!blob) { alert('Capture failed. Please try again.'); return; }
    const file = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
    setFile(file);
    closeCamera();
  }, 'image/jpeg', 0.92);
}

// ── Analyze flow ────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  uploadSection.hidden  = true;
  resultsSection.hidden = true;
  loadingSection.hidden = false;
  analyzeBtn.disabled   = true;

  try {
    // 1 — analyze
    setStatus('Analyzing face…');
    const fd1 = new FormData();
    fd1.append('file', selectedFile);
    const aRes = await fetch('/analyze', { method: 'POST', body: fd1 });
    if (!aRes.ok) {
      const err = await aRes.json().catch(() => ({}));
      throw new Error(err.detail || 'Analysis failed');
    }
    analysis = await aRes.json();

    // 2 — recommend
    setStatus('Detecting skin tone…');
    const rRes = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(analysis),
    });
    if (!rRes.ok) throw new Error('Recommendation request failed');
    recommend = await rRes.json();

    // 3 — show results
    setStatus('Generating recommendations…');
    await delay(600);

    renderResults(analysis, recommend);
    loadingSection.hidden = true;
    resultsSection.hidden = false;

    // 4 — generate outfit images (non-blocking)
    generateImages(analysis, recommend);

  } catch (err) {
    loadingSection.hidden = true;
    uploadSection.hidden  = false;
    analyzeBtn.disabled   = false;
    alert('Error: ' + err.message);
  }
});

const setStatus = text => { loadingText.textContent = text; };
const delay     = ms   => new Promise(r => setTimeout(r, ms));

// ── Render results ──────────────────────────────────────────
function renderResults(a, r) {
  profileItems.innerHTML = [
    ['Gender',     a.gender],
    ['Skin Tone',  a.skin_tone],
    ['Face Shape', a.face_shape],
  ].map(([label, value]) => `
    <div class="profile-row">
      <span class="row-label">${label}</span>
      <span class="row-value">${value}</span>
    </div>`).join('');

  swatchesEl.innerHTML = r.colors.slice(0, 6).map(c => `
    <div class="swatch">
      <div class="dot" style="background:${COLOR_CSS[c] || '#ccc'}"></div>
      <span class="name">${c}</span>
    </div>`).join('');

  outfitTags.innerHTML = r.garments.slice(0, 6).map(g => {
    const detail = (r.garments_detail || []).find(d => d.name === g);
    const occ    = detail ? detail.occasion : '';
    return `<span class="tag">${g}${occ ? `<span class="tag-sub">${occ}</span>` : ''}</span>`;
  }).join('');

  accessoryTags.innerHTML = (r.accessories || [])
    .map(a => `<span class="tag">${a}</span>`).join('');

  styleDesc.textContent = r.style || '';
  tipsList.innerHTML    = (r.style_tips || []).map(t => `<li>${t}</li>`).join('');

  genStatus.style.display = 'flex';
  genImages.innerHTML     = '';
}

// ── Image generation — sends user's photo for virtual try-on ─
async function generateImages(a, r) {
  try {
    // Send the actual photo so the model dresses the same person
    const fd = new FormData();
    fd.append('file',      selectedFile);
    fd.append('gender',    a.gender);
    fd.append('skin_tone', a.skin_tone);
    fd.append('garments',  JSON.stringify(r.garments));
    fd.append('colors',    JSON.stringify(r.colors));
    fd.append('style',     r.style);

    const res = await fetch('/generate', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Generation failed');
    }

    const data = await res.json();
    genStatus.style.display = 'none';

    if (data.images && data.images.length) {
      genImages.innerHTML = data.images
        .map(src => `<img src="${src}" alt="You in this outfit" />`)
        .join('');
    } else {
      genImages.innerHTML = '<p class="gen-error">No images returned from API.</p>';
    }
  } catch (err) {
    genStatus.style.display = 'none';
    genImages.innerHTML = `<p class="gen-error">Image generation: ${err.message}</p>`;
  }
}
