/* ─── DOM references ────────────────────────────────────────────────── */
const form            = document.querySelector("#analysis-form");
const photoInput      = document.querySelector("#photo");
const fileName        = document.querySelector("#file-name");
const analyzeButton   = document.querySelector("#analyze-button");
const statusMessage   = document.querySelector("#status-message");
const results         = document.querySelector("#results");
const pageNavigation  = document.querySelector(".page-nav");
const regionSelect    = document.querySelector("#region");
const targetColorSelect = document.querySelector("#target-color");
const cameraToggle    = document.querySelector("#camera-toggle");
const cameraPanel     = document.querySelector("#camera-panel");
const cameraVideo     = document.querySelector("#camera-video");
const cameraStatus    = document.querySelector("#camera-status");

/* ─── State ─────────────────────────────────────────────────────────── */
let imageUrl      = null;
let sourceImage   = null;
let lastPayload   = null;
let cameraStream  = null;

/* ─── Colour palette (fallback before API load) ─────────────────────── */
const shadeColors = {
  "Soft Black": "#171717", "Jet Black": "#101010", "Mocha Brown": "#4C2C2A",
  "Natural Black": "#211B18", Mahogany: "#6E2C2C", "Warm Brown": "#7A4B34",
  Espresso: "#3B241C", "Chocolate Brown": "#5A3825", Chestnut: "#7A4A2B",
  Cinnamon: "#A55A36", Auburn: "#8A3324", "Rust Copper": "#B86135",
  Copper: "#B65E32", "Rose Gold": "#C9806E", Burgundy: "#6D1F3A",
  "Red Violet": "#8A3A5C", Caramel: "#B97943", "Sunlit Blonde": "#D9B77A",
  "Honey Blonde": "#C99A52", "Strawberry Blonde": "#D88C6B",
  "Golden Blonde": "#D6B36A", "Platinum Blonde": "#E7E1D6",
  "Silver Ash": "#C5C8D0", "Ash Blonde": "#B7AA91", "Icy Blonde": "#DDEAF1",
};

/* ─── Load palette from API ─────────────────────────────────────────── */
async function loadSharedColorPalette() {
  try {
    const response = await fetch("/api/v1/overview");
    if (!response.ok) return;
    const overview = await response.json();
    for (const shade of overview.color_palette || []) {
      shadeColors[shade.name] = shade.hex;
      if (![...targetColorSelect.options].some(o => o.value === shade.name)) {
        targetColorSelect.add(new Option(shade.name, shade.name));
      }
    }
    updateShadeChoice();
  } catch {
    /* palette will use the built-in fallback */
  }
}

/* ─── Render haircut recommendation cards (with image) ──────────────── */
function renderHaircutRecommendations(recommendations = []) {
  const container = document.querySelector("#haircut-recommendations");
  container.replaceChildren();
  recommendations.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "haircut-option";

    /* photo */
    if (item.image) {
      const img = document.createElement("img");
      img.src = item.image;
      img.alt = item.name;
      img.className = "haircut-image";
      img.loading = "lazy";
      card.append(img);
    }

    /* info row */
    const info = document.createElement("div");
    info.className = "haircut-info";

    const number = document.createElement("span");
    number.className = "haircut-number";
    number.textContent = String(index + 1).padStart(2, "0");

    const name = document.createElement("strong");
    name.textContent = item.name;

    const description = document.createElement("p");
    description.textContent = item.description;

    info.append(number, name, description);
    card.append(info);
    container.append(card);
  });
}

/* ─── Render colour recommendation chips ────────────────────────────── */
function renderColorRecommendations(recommendations = []) {
  const container = document.querySelector("#color-recommendations");
  container.replaceChildren();
  for (const shade of recommendations) {
    const button = document.createElement("button");
    const swatch = document.createElement("span");
    const details = document.createElement("span");
    const name = document.createElement("strong");
    const description = document.createElement("small");
    button.type = "button";
    button.className = "shade-recommendation";
    swatch.className = "shade-recommendation-swatch";
    swatch.style.backgroundColor = shade.hex;
    details.className = "shade-recommendation-copy";
    name.textContent = shade.name;
    description.textContent = shade.description;
    details.append(name, description);
    button.append(swatch, details);
    button.addEventListener("click", () => {
      targetColorSelect.value = shade.name;
      updateShadeChoice();
      renderColorPreview();
    });
    container.append(button);
  }
}

/* ─── Shade swatch preview ───────────────────────────────────────────── */
function updateShadeChoice() {
  const shade = targetColorSelect.value;
  document.querySelector("#shade-swatch").style.backgroundColor = shadeColors[shade] || "transparent";
  document.querySelector("#shade-name").textContent = shade || "No shade selected";
}

/* ─── Page navigation ────────────────────────────────────────────────── */
function showPage(page) {
  for (const section of document.querySelectorAll(".results-page")) {
    const isActive = section.id === `page-${page}`;
    section.hidden = !isActive;
    section.classList.toggle("is-active", isActive);
  }
  for (const button of pageNavigation.querySelectorAll("[data-page]")) {
    const isActive = button.dataset.page === page;
    button.classList.toggle("is-active", isActive);
    if (isActive) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
}
pageNavigation.addEventListener("click", (event) => {
  const button = event.target.closest("[data-page]");
  if (button) showPage(button.dataset.page);
});

/* ═══════════════════════════════════════════════════════════════════════
   CAMERA — fixed implementation
   ═══════════════════════════════════════════════════════════════════════ */

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraPanel.hidden = true;
  cameraToggle.disabled = false;
  cameraToggle.textContent = "Use camera";
  cameraStatus.textContent = "";
}

cameraToggle.addEventListener("click", async () => {
  /* If camera is already on, just turn it off */
  if (cameraStream) {
    stopCamera();
    return;
  }

  /* Check API availability */
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraStatus.textContent =
      "Camera access is unavailable in this browser or the page is not served over HTTPS/localhost. Please upload an image instead.";
    cameraPanel.hidden = false;
    return;
  }

  cameraToggle.disabled = true;
  cameraPanel.hidden = false;
  cameraStatus.textContent = "Requesting camera permission…";

  try {
    /* Use a simple video-only constraint — no facingMode — for broad desktop/mobile compatibility */
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraStream = stream;
    cameraVideo.srcObject = stream;

    /* play() is a Promise; catch autoplay-policy errors */
    try {
      await cameraVideo.play();
    } catch (playError) {
      /* Some browsers need a user gesture for play(); the video should
         still display live because srcObject is set. Ignore the error. */
    }

    cameraToggle.disabled = false;
    cameraToggle.textContent = "Turn camera off";
    cameraStatus.textContent = "Frame your hair clearly, then press Capture photo.";
  } catch (error) {
    cameraStream = null;
    cameraVideo.srcObject = null;
    cameraToggle.disabled = false;

    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      cameraStatus.textContent =
        "Camera permission was denied. Allow camera access in your browser settings and reload the page.";
    } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      cameraStatus.textContent =
        "No camera was found on this device. Please upload an image instead.";
    } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
      cameraStatus.textContent =
        "The camera is in use by another application. Close it and try again.";
    } else {
      cameraStatus.textContent = `Could not start the camera: ${error.message || error.name}`;
    }
  }
});

document.querySelector("#close-camera").addEventListener("click", stopCamera);

document.querySelector("#capture-photo").addEventListener("click", () => {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
    cameraStatus.textContent = "The camera is still starting. Wait a moment and try again.";
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.width  = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  canvas.getContext("2d").drawImage(cameraVideo, 0, 0);

  canvas.toBlob((blob) => {
    if (!blob) {
      cameraStatus.textContent = "Could not capture the frame. Try again.";
      return;
    }
    const file = new File([blob], `hair-photo-${Date.now()}.jpg`, { type: "image/jpeg" });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    photoInput.files = transfer.files;

    /* Trigger the change handler (updates preview, resets results) */
    photoInput.dispatchEvent(new Event("change", { bubbles: true }));

    /* Stop camera after capture */
    stopCamera();
  }, "image/jpeg", 0.94);
});

/* ─── File / photo-input change ──────────────────────────────────────── */
photoInput.addEventListener("change", () => {
  const file = photoInput.files[0];
  fileName.textContent = file ? file.name : "JPG, PNG, or WEBP";
  results.hidden = true;
  statusMessage.hidden = true;
  showPage("type");
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = file ? URL.createObjectURL(file) : null;
});

regionSelect.addEventListener("change", renderSelectedRegion);
targetColorSelect.addEventListener("change", () => {
  updateShadeChoice();
  if (lastPayload && sourceImage) renderColorPreview();
});

updateShadeChoice();
loadSharedColorPalette();

/* ─── Form submit → analysis ─────────────────────────────────────────── */
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = photoInput.files[0];
  if (!file) return showStatus("Choose an image before starting the analysis.", true);
  const isFirstAnalysis = results.hidden;

  const request = new FormData();
  request.append("file", file);
  const parameters = new URLSearchParams({ region: regionSelect.value });
  const targetColor = targetColorSelect.value;
  if (targetColor) parameters.set("target_color", targetColor);

  analyzeButton.disabled = true;
  analyzeButton.firstElementChild.textContent = "Analyzing image…";
  showStatus("Running the image analysis. This may take a moment.");

  try {
    const response = await fetch(`/api/v1/analyze?${parameters}`, { method: "POST", body: request });
    const payload  = await response.json();
    if (!response.ok) {
      const detail = payload.detail || "The analysis request failed.";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    await renderResults(payload, file);
    results.hidden = false;
    statusMessage.hidden = true;
    if (isFirstAnalysis) showPage("type");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showStatus(error.message || "The analysis could not be completed.", true);
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.firstElementChild.textContent = "Run analysis";
  }
});

/* ─── Helpers ────────────────────────────────────────────────────────── */
function showStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("error", isError);
  statusMessage.hidden = false;
}

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function setCandidates(selector, candidates = []) {
  const list = document.querySelector(selector);
  list.replaceChildren();
  for (const candidate of candidates) {
    const item       = document.createElement("li");
    const label      = document.createElement("span");
    const confidence = document.createElement("span");
    label.textContent      = candidate.label;
    confidence.textContent = `${Number(candidate.confidence).toFixed(1)}%`;
    item.append(label, confidence);
    list.append(item);
  }
}

/* ─── Render all analysis results ────────────────────────────────────── */
async function renderResults(payload, file) {
  lastPayload = payload;
  const quality = payload.quality || {};
  setText("#hair-type",              payload.hair_type || "Unknown");
  setText("#hair-type-confidence",   `Model confidence · ${Number(payload.hair_type_confidence || 0).toFixed(1)}%`);
  setText("#all-hair-type",          payload.hair_type || "Unknown");
  setText("#all-hair-type-confidence", `Model confidence · ${Number(payload.hair_type_confidence || 0).toFixed(1)}%`);
  setText("#disease",                payload.disease || "Unknown");
  setText("#disease-confidence",     `Validation-set model score · ${Number(payload.disease_confidence || 0).toFixed(1)}%`);
  setText("#all-disease",            payload.disease || "Unknown");
  setText("#all-disease-confidence", `Validation-set model score · ${Number(payload.disease_confidence || 0).toFixed(1)}%`);
  setText("#source-name",            `${file.name} · ${(file.size / 1024).toFixed(0)} KB`);
  setText("#coverage",               `${Number(payload.coverage || 0).toFixed(1)}%`);
  setText("#quality-score",          `${Number(quality.score || 0).toFixed(0)} / 100`);
  setText("#quality-status",         quality.status || "Not available");

  const selectedRegion = payload.current_region || regionSelect.value;
  regionSelect.value = selectedRegion;
  targetColorSelect.value = payload.target_color || targetColorSelect.value;
  updateShadeChoice();

  setCandidates("#hair-type-candidates", payload.hair_type_top_predictions);
  setCandidates("#disease-candidates",   payload.disease_top_predictions);
  setCandidates("#condition-shortlist",  (payload.disease_top_predictions || []).slice(0, 3));
  renderHaircutRecommendations(payload.haircut_recommendations);
  renderColorRecommendations(payload.recommended_colors);

  sourceImage = new Image();
  sourceImage.src = imageUrl;
  await sourceImage.decode();
  renderSelectedRegion();
}

/* ─── Region / colour helpers ────────────────────────────────────────── */
function selectedRegionColor() {
  return lastPayload?.region_colors?.[regionSelect.value] || lastPayload?.color || {};
}

function renderSelectedRegion() {
  if (!lastPayload || !sourceImage) return;
  const region      = regionSelect.value;
  const mask        = lastPayload.region_masks?.[region] || lastPayload.hair_mask || [];
  const regionColor = selectedRegionColor();

  setText("#result-region",    region);
  setText("#hair-map-title",   `${region} region`);
  setText("#hair-map-caption", `${region} region · ${Number(lastPayload.regions?.[region] || 0).toFixed(1)}% of image area`);
  setText("#color-label",      regionColor.label || "Estimated hair color");
  setText("#color-value",      `${regionColor.hex || "#505050"} · ${(regionColor.rgb || [80, 80, 80]).join(", ")}`);
  setText("#color-region",     `Estimated in the ${region} region.`);
  setText("#all-color",        regionColor.label || "Unknown");
  setText("#all-color-value",  regionColor.hex || "");
  setText("#all-target-color", targetColorSelect.value || "Not selected");
  setText("#all-selected-region", region);

  document.querySelector("#color-swatch").style.backgroundColor = regionColor.hex || "#505050";
  drawHairMask(sourceImage, mask);
  renderColorPreview();
}

/* ─── Hair-mask canvas overlay ───────────────────────────────────────── */
function scaledCanvasSize(mask) {
  const sourceHeight = mask.length;
  const sourceWidth  = mask[0].length;
  const scale = Math.min(1, 1400 / Math.max(sourceWidth, sourceHeight));
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
    sourceWidth,
    sourceHeight,
  };
}

function drawHairMask(image, mask) {
  if (!mask.length || !mask[0]?.length) return;
  const { width, height, sourceWidth, sourceHeight } = scaledCanvasSize(mask);
  const canvas  = document.querySelector("#hair-map");
  canvas.width  = width;
  canvas.height = height;
  const ctx    = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height);
  const tint   = [196, 84, 63];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[Math.floor(y * sourceHeight / height)][Math.floor(x * sourceWidth / width)]) continue;
      const i = (y * width + x) * 4;
      pixels.data[i]     = pixels.data[i]     * 0.58 + tint[0] * 0.42;
      pixels.data[i + 1] = pixels.data[i + 1] * 0.58 + tint[1] * 0.42;
      pixels.data[i + 2] = pixels.data[i + 2] * 0.58 + tint[2] * 0.42;
    }
  }
  ctx.putImageData(pixels, 0, 0);
}

/* ─── Colour math ────────────────────────────────────────────────────── */
function hexToRgb(hex) {
  return [1, 3, 5].map(offset => Number.parseInt(hex.slice(offset, offset + 2), 16));
}

function rgbToHsv(r, g, b) {
  const [rv, gv, bv] = [r / 255, g / 255, b / 255];
  const max = Math.max(rv, gv, bv);
  const min = Math.min(rv, gv, bv);
  const delta = max - min;
  let hue = 0;
  if (delta) {
    if (max === rv) hue = 60 * (((gv - bv) / delta) % 6);
    else if (max === gv) hue = 60 * ((bv - rv) / delta + 2);
    else hue = 60 * ((rv - gv) / delta + 4);
  }
  return [((hue + 360) % 360), max ? delta / max : 0, max];
}

function hsvToRgb(h, s, v) {
  const c = v * s;
  const seg = h / 60;
  const x = c * (1 - Math.abs(seg % 2 - 1));
  const m = v - c;
  let rgb;
  if (seg < 1) rgb = [c, x, 0];
  else if (seg < 2) rgb = [x, c, 0];
  else if (seg < 3) rgb = [0, c, x];
  else if (seg < 4) rgb = [0, x, c];
  else if (seg < 5) rgb = [x, 0, c];
  else rgb = [c, 0, x];
  return rgb.map(ch => Math.round((ch + m) * 255));
}

/* ─── Colour preview canvas ──────────────────────────────────────────── */
function renderColorPreview() {
  if (!lastPayload || !sourceImage) return;
  const targetHex = shadeColors[targetColorSelect.value];
  const mask = lastPayload.region_masks?.[regionSelect.value] || lastPayload.hair_mask || [];
  if (!targetHex || !mask.length || !mask[0]?.length) return;

  const { width, height, sourceWidth, sourceHeight } = scaledCanvasSize(mask);

  /* Draw source image */
  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = width;
  srcCanvas.height = height;
  const srcCtx = srcCanvas.getContext("2d", { willReadFrequently: true });
  srcCtx.drawImage(sourceImage, 0, 0, width, height);
  const imageData = srcCtx.getImageData(0, 0, width, height);

  /* Build white mask canvas */
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  const maskPixels = maskCtx.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[Math.floor(y * sourceHeight / height)][Math.floor(x * sourceWidth / width)]) continue;
      const i = (y * width + x) * 4;
      maskPixels.data[i] = maskPixels.data[i + 1] = maskPixels.data[i + 2] = maskPixels.data[i + 3] = 255;
    }
  }
  maskCtx.putImageData(maskPixels, 0, 0);

  /* Soft (blurred) mask for feathered edges */
  const softCanvas = document.createElement("canvas");
  softCanvas.width = width;
  softCanvas.height = height;
  const softCtx = softCanvas.getContext("2d", { willReadFrequently: true });
  softCtx.filter = "blur(2px)";
  softCtx.drawImage(maskCanvas, 0, 0);
  const alphaData = softCtx.getImageData(0, 0, width, height).data;

  const [tr, tg, tb] = hexToRgb(targetHex);
  const [th, ts, tv] = rgbToHsv(tr, tg, tb);
  const strength         = 0.78;
  const brightnessStrength = strength * 0.58;

  for (let i = 0; i < imageData.data.length; i += 4) {
    const alpha = (alphaData[i + 3] / 255) * strength;
    if (alpha <= 0.005) continue;
    const [h, s, v] = rgbToHsv(imageData.data[i], imageData.data[i + 1], imageData.data[i + 2]);
    const recolored = hsvToRgb(
      th,
      Math.min(1, s * (1 - strength) + ts * strength + (25 / 255) * strength),
      v * (1 - brightnessStrength) + tv * brightnessStrength,
    );
    for (let ch = 0; ch < 3; ch++) {
      imageData.data[i + ch] = Math.round(imageData.data[i + ch] * (1 - alpha) + recolored[ch] * alpha);
    }
  }

  srcCtx.putImageData(imageData, 0, 0);
  document.querySelector("#original-preview").src = imageUrl;
  const preview = document.querySelector("#preview-image");
  preview.src = srcCanvas.toDataURL("image/jpeg", 0.92);
  preview.alt = `${targetColorSelect.value} virtual hair colour preview`;
  setText("#preview-caption", `Virtual preview · ${targetColorSelect.value}`);
  document.querySelector("#preview-comparison").hidden = false;
}