const state = {
  alignmentReady: false,
  colorCalibration: null,
  cameras: {
    camera1: createCameraState(),
    camera2: createCameraState(),
  },
};

function createCameraState() {
  return { imageId: null, image: null, points: [], dragging: -1, calibrated: false, calibration: null, rotationDegrees: 0, flipHorizontal: false };
}

const lengthInput = document.querySelector("#pool-length");
const widthInput = document.querySelector("#pool-width");
const saveButton = document.querySelector("#save-config");
const saveMessage = document.querySelector("#save-message");
const template = document.querySelector("#camera-template");
const cameraList = document.querySelector("#camera-list");
const alignmentButton = document.querySelector("#generate-alignment");
const alignmentMessage = document.querySelector("#alignment-message");
const captureAllButton = document.querySelector("#capture-all");
const captureMessage = document.querySelector("#capture-message");

for (const [index, cameraId] of ["camera1", "camera2"].entries()) {
  const fragment = template.content.cloneNode(true);
  const card = fragment.querySelector(".camera-card");
  card.dataset.cameraId = cameraId;
  card.querySelector(".camera-number").textContent = `0${index + 1}`;
  card.querySelector("h3").textContent = `카메라 ${index + 1}`;
  cameraList.appendChild(fragment);
  bindCameraCard(cameraId, cameraList.lastElementChild);
}

lengthInput.addEventListener("input", dimensionsChanged);
widthInput.addEventListener("input", dimensionsChanged);
saveButton.addEventListener("click", saveConfig);
alignmentButton.addEventListener("click", generateAlignmentPreview);
captureAllButton.addEventListener("click", captureAllCameras);
loadStartupCaptures();

function bindCameraCard(cameraId, card) {
  const input = card.querySelector(".image-input");
  const canvas = card.querySelector(".point-canvas");
  card.querySelector(".choose-image").addEventListener("click", () => input.click());
  card.querySelector(".change-image").addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files[0] && uploadImage(cameraId, card, input.files[0]));
  card.querySelector(".reset-points").addEventListener("click", () => resetPoints(cameraId, card));
  card.querySelector(".preview-button").addEventListener("click", () => preview(cameraId, card));
  card.querySelector(".rotate-left").addEventListener("click", () => rotatePreview(cameraId, card, -90));
  card.querySelector(".rotate-right").addEventListener("click", () => rotatePreview(cameraId, card, 90));
  card.querySelector(".flip-horizontal").addEventListener("click", () => toggleHorizontalFlip(cameraId, card));
  canvas.addEventListener("pointerdown", event => pointerDown(event, cameraId, card));
  canvas.addEventListener("pointermove", event => pointerMove(event, cameraId, card));
  canvas.addEventListener("pointerup", event => pointerUp(event, cameraId, card));
  canvas.addEventListener("pointercancel", event => pointerUp(event, cameraId, card));
}

async function uploadImage(cameraId, card, file) {
  clearError(card);
  if (file.size > 16 * 1024 * 1024) return showError(card, "이미지 크기는 16MB 이하여야 합니다.");
  const form = new FormData();
  form.append("camera_id", cameraId);
  form.append("image", file);
  setCardState(card, "업로드 중…");
  try {
    const response = await fetch("/api/images", { method: "POST", body: form });
    const data = await readResponse(response);
    await applyCameraImage(cameraId, card, data);
  } catch (error) {
    setCardState(card, "이미지 필요");
    showError(card, error.message);
  }
}

function applyCameraImage(cameraId, card, data) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      invalidateAlignment();
      state.cameras[cameraId] = { ...createCameraState(), imageId: data.image_id, image };
      card.querySelector(".upload-zone").classList.add("has-image");
      card.querySelector(".canvas-toolbar").hidden = false;
      card.querySelector(".preview-image").hidden = true;
      card.querySelector(".preview-empty").hidden = false;
      card.querySelector(".preview-meta").hidden = true;
      card.querySelector(".preview-controls").hidden = true;
      setCardState(card, "점 지정 중");
      resizeAndDraw(cameraId, card);
      updatePrompt(cameraId, card);
      updateCompletion();
      resolve();
    };
    image.onerror = () => reject(new Error(`${cameraId === "camera1" ? "카메라 1" : "카메라 2"} 이미지를 불러오지 못했습니다.`));
    image.src = `${data.image_url}&v=${Date.now()}`;
  });
}

async function loadStartupCaptures() {
  try {
    const response = await fetch("/api/cameras/latest", { cache: "no-store" });
    if (!response.ok) {
      for (const cameraId of ["camera1", "camera2"]) {
        setCardState(cameraList.querySelector(`[data-camera-id="${cameraId}"]`), "이미지 필요");
      }
      captureMessage.textContent = "자동 촬영 이미지가 없으면 직접 업로드하거나 다시 촬영하세요.";
      return;
    }
    const data = await response.json();
    for (const item of data.cameras) {
      const card = cameraList.querySelector(`[data-camera-id="${item.camera_id}"]`);
      await applyCameraImage(item.camera_id, card, item);
    }
    captureMessage.textContent = "부팅 때 자동 촬영된 카메라 1·2 이미지를 불러왔습니다.";
  } catch (_error) {
    captureMessage.textContent = "자동 촬영 이미지를 확인하지 못했습니다. 직접 업로드할 수 있습니다.";
  }
}

async function captureAllCameras() {
  captureAllButton.disabled = true;
  captureMessage.classList.remove("error");
  captureMessage.textContent = "카메라 1·2를 차례로 촬영하고 있습니다…";
  for (const cameraId of ["camera1", "camera2"]) {
    clearError(cameraList.querySelector(`[data-camera-id="${cameraId}"]`));
    setCardState(cameraList.querySelector(`[data-camera-id="${cameraId}"]`), "촬영 중…");
  }
  try {
    const response = await fetch("/api/cameras/capture", { method: "POST" });
    const data = await readResponse(response);
    for (const item of data.cameras) {
      const card = cameraList.querySelector(`[data-camera-id="${item.camera_id}"]`);
      await applyCameraImage(item.camera_id, card, item);
    }
    captureMessage.textContent = "새 이미지를 촬영해 점 지정 영역에 올렸습니다.";
    showToast("카메라 1·2 촬영 완료");
  } catch (error) {
    captureMessage.classList.add("error");
    captureMessage.textContent = error.message;
    for (const cameraId of ["camera1", "camera2"]) {
      const camera = state.cameras[cameraId];
      const label = !camera.image ? "이미지 필요" : camera.calibrated ? "보정 완료" : "점 지정 중";
      setCardState(cameraList.querySelector(`[data-camera-id="${cameraId}"]`), label, camera.calibrated);
    }
    updateCompletion();
  } finally {
    captureAllButton.disabled = false;
  }
}

function dimensionsChanged() {
  invalidateAlignment();
  document.querySelector("#dimension-error").textContent = validateDimensions() || "";
  for (const [cameraId, camera] of Object.entries(state.cameras)) {
    if (camera.calibrated) {
      camera.calibrated = false;
      camera.calibration = null;
      const card = cameraList.querySelector(`[data-camera-id="${cameraId}"]`);
      setCardState(card, camera.points.length === 4 ? "미리보기 필요" : "점 지정 중");
      card.querySelector(".preview-controls").hidden = true;
    }
  }
  updateCompletion();
}

function validateDimensions() {
  const length = Number(lengthInput.value);
  const width = Number(widthInput.value);
  if (!Number.isFinite(length) || !Number.isFinite(width)) return "길이와 폭을 숫자로 입력해 주세요.";
  if (length < 2 || length > 100) return "길이는 2~100m 범위로 입력해 주세요.";
  if (width < 2 || width > 50) return "폭은 2~50m 범위로 입력해 주세요.";
  return null;
}

function pointerDown(event, cameraId, card) {
  const camera = state.cameras[cameraId];
  if (!camera.image) return;
  event.currentTarget.setPointerCapture(event.pointerId);
  const point = naturalPoint(event, camera);
  const hit = findPoint(camera, point, event.currentTarget);
  if (hit >= 0) camera.dragging = hit;
  else if (camera.points.length < 4) {
    camera.points.push(point);
    if (camera.points.length === 4) camera.points = orderPointsByImage(camera.points);
    invalidateCalibration(cameraId, card);
    draw(cameraId, card);
    updatePrompt(cameraId, card);
  }
}

function pointerMove(event, cameraId, card) {
  const camera = state.cameras[cameraId];
  if (camera.dragging < 0) return;
  camera.points[camera.dragging] = naturalPoint(event, camera);
  invalidateCalibration(cameraId, card);
  draw(cameraId, card);
}

function pointerUp(event, cameraId, card) {
  const camera = state.cameras[cameraId];
  if (camera.dragging >= 0) {
    camera.dragging = -1;
    if (camera.points.length === 4) camera.points = orderPointsByImage(camera.points);
    draw(cameraId, card);
    updatePrompt(cameraId, card);
  }
  if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
}

function naturalPoint(event, camera) {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width * camera.image.naturalWidth, 0, camera.image.naturalWidth - 1),
    y: clamp((event.clientY - rect.top) / rect.height * camera.image.naturalHeight, 0, camera.image.naturalHeight - 1),
  };
}

function findPoint(camera, target, canvas) {
  const hitRadius = 24 * camera.image.naturalWidth / canvas.getBoundingClientRect().width;
  return camera.points.findIndex(point => Math.hypot(point.x - target.x, point.y - target.y) <= hitRadius);
}

function orderPointsByImage(points) {
  const entries = points.map((point, index) => ({ point, index, sum: point.x + point.y, difference: point.y - point.x }));
  const tl = entries.reduce((best, entry) => entry.sum < best.sum ? entry : best);
  const tr = entries.reduce((best, entry) => entry.difference < best.difference ? entry : best);
  const br = entries.reduce((best, entry) => entry.sum > best.sum ? entry : best);
  const bl = entries.reduce((best, entry) => entry.difference > best.difference ? entry : best);
  const ordered = [tl, tr, br, bl];
  if (new Set(ordered.map(entry => entry.index)).size === 4) return ordered.map(entry => entry.point);

  const byY = [...points].sort((a, b) => a.y - b.y);
  const top = byY.slice(0, 2).sort((a, b) => a.x - b.x);
  const bottom = byY.slice(2).sort((a, b) => a.x - b.x);
  return [top[0], top[1], bottom[1], bottom[0]];
}

function resetPoints(cameraId, card) {
  const camera = state.cameras[cameraId];
  camera.points = [];
  invalidateCalibration(cameraId, card);
  draw(cameraId, card);
  updatePrompt(cameraId, card);
}

function invalidateCalibration(cameraId, card) {
  invalidateAlignment();
  const camera = state.cameras[cameraId];
  camera.calibrated = false;
  camera.calibration = null;
  card.querySelector(".preview-controls").hidden = true;
  card.querySelector(".preview-button").disabled = camera.points.length !== 4;
  setCardState(card, camera.points.length === 4 ? "미리보기 필요" : "점 지정 중");
  updateCompletion();
}

function resizeAndDraw(cameraId, card) {
  const camera = state.cameras[cameraId];
  const canvas = card.querySelector(".point-canvas");
  canvas.width = camera.image.naturalWidth;
  canvas.height = camera.image.naturalHeight;
  draw(cameraId, card);
}

function draw(cameraId, card) {
  const camera = state.cameras[cameraId];
  const canvas = card.querySelector(".point-canvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(camera.image, 0, 0, canvas.width, canvas.height);
  if (!camera.points.length) return;

  const scale = canvas.width / Math.max(canvas.clientWidth, 1);
  context.lineWidth = 3 * scale;
  context.strokeStyle = "#42d6c5";
  context.fillStyle = "rgba(66, 214, 197, .16)";
  context.beginPath();
  context.moveTo(camera.points[0].x, camera.points[0].y);
  camera.points.slice(1).forEach(point => context.lineTo(point.x, point.y));
  if (camera.points.length === 4) context.closePath();
  context.fill();
  context.stroke();

  camera.points.forEach((point, index) => {
    const radius = (index === camera.dragging ? 6 : 4.5) * scale;
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = "#ff3b30";
    context.fill();
    context.lineWidth = 1.5 * scale;
    context.strokeStyle = "#ffffff";
    context.stroke();

    const labelX = point.x + 11 * scale;
    const labelY = point.y - 10 * scale;
    context.font = `800 ${10 * scale}px system-ui`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.lineWidth = 3.5 * scale;
    context.strokeStyle = "rgba(7, 27, 38, .92)";
    context.strokeText(String(index + 1), labelX, labelY);
    context.fillStyle = "#ffffff";
    context.fillText(String(index + 1), labelX, labelY);
  });
}

function updatePrompt(cameraId, card) {
  const count = state.cameras[cameraId].points.length;
  card.querySelector(".point-prompt").textContent = count < 4
    ? `수영장 모서리를 눌러주세요 · ${count}/4`
    : "이미지 기준으로 자동 정렬됨 · 점을 드래그해 미세 조정하세요";
  card.querySelector(".preview-button").disabled = count !== 4;
}

async function preview(cameraId, card, adjustment = null) {
  invalidateAlignment();
  clearError(card);
  const dimensionError = validateDimensions();
  if (dimensionError) {
    document.querySelector("#dimension-error").textContent = dimensionError;
    lengthInput.focus();
    return;
  }
  const button = card.querySelector(".preview-button");
  button.disabled = true;
  button.querySelector("b").textContent = "···";
  try {
    const response = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFor(cameraId)),
    });
    const data = await readResponse(response);
    const camera = state.cameras[cameraId];
    camera.calibrated = true;
    camera.calibration = data.calibration;
    const previewImage = card.querySelector(".preview-image");
    previewImage.src = `${data.preview_url}?v=${Date.now()}`;
    previewImage.hidden = false;
    card.querySelector(".preview-empty").hidden = true;
    card.querySelector(".preview-meta").hidden = false;
    card.querySelector(".preview-controls").hidden = false;
    card.querySelector(".rotation-value").textContent = `${camera.rotationDegrees}°`;
    const flipButton = card.querySelector(".flip-horizontal");
    flipButton.classList.toggle("active", camera.flipHorizontal);
    flipButton.querySelector("span").textContent = camera.flipHorizontal ? "반전 해제" : "좌우 반전";
    const orientedSize = data.calibration.oriented_coordinate_size_m;
    card.querySelector(".preview-size").textContent = `${orientedSize.width}m × ${orientedSize.height}m`;
    setCardState(card, "보정 완료", true);
    const toastMessage = adjustment === "rotation"
      ? `방향을 ${camera.rotationDegrees}°로 적용했습니다.`
      : adjustment === "flip"
        ? `좌우 반전을 ${camera.flipHorizontal ? "적용" : "해제"}했습니다.`
        : `${card.querySelector("h3").textContent} 미리보기가 만들어졌습니다.`;
    showToast(toastMessage);
  } catch (error) {
    showError(card, error.message);
    setCardState(card, "점 확인 필요");
  } finally {
    button.disabled = false;
    button.querySelector("b").textContent = "→";
    updateCompletion();
  }
}

async function rotatePreview(cameraId, card, delta) {
  const camera = state.cameras[cameraId];
  camera.rotationDegrees = (camera.rotationDegrees + delta + 360) % 360;
  const controls = card.querySelectorAll(".preview-controls button");
  controls.forEach(button => button.disabled = true);
  try {
    await preview(cameraId, card, "rotation");
  } finally {
    controls.forEach(button => button.disabled = false);
  }
}

async function toggleHorizontalFlip(cameraId, card) {
  const camera = state.cameras[cameraId];
  camera.flipHorizontal = !camera.flipHorizontal;
  const controls = card.querySelectorAll(".preview-controls button");
  controls.forEach(button => button.disabled = true);
  try {
    await preview(cameraId, card, "flip");
  } finally {
    controls.forEach(button => button.disabled = false);
  }
}

function invalidateAlignment() {
  state.alignmentReady = false;
  state.colorCalibration = null;
  const image = document.querySelector("#alignment-image");
  if (image) image.hidden = true;
  const empty = document.querySelector("#alignment-empty");
  if (empty) empty.hidden = false;
  const meta = document.querySelector("#alignment-meta");
  if (meta) meta.hidden = true;
  renderColorCalibration(null);
}

async function generateAlignmentPreview() {
  alignmentButton.disabled = true;
  alignmentButton.querySelector("b").textContent = "···";
  alignmentMessage.classList.remove("error");
  alignmentMessage.textContent = "두 카메라 좌표를 겹치고 있습니다…";
  try {
    const response = await fetch("/api/alignment-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pool: { length_m: Number(lengthInput.value), width_m: Number(widthInput.value) },
        cameras: [payloadFor("camera1"), payloadFor("camera2")],
      }),
    });
    const data = await readResponse(response);
    document.querySelector("#alignment-image").src = `${data.preview_url}?v=${Date.now()}`;
    document.querySelector("#alignment-image").hidden = false;
    document.querySelector("#alignment-empty").hidden = true;
    document.querySelector("#alignment-meta").hidden = false;
    document.querySelector("#alignment-size").textContent = `${data.coordinate_size_m.width}m × ${data.coordinate_size_m.height}m · 격자 ${data.grid_step_m}m`;
    state.colorCalibration = data.color_calibration;
    renderColorCalibration(data.color_calibration);
    state.alignmentReady = true;
    alignmentMessage.textContent = "XY 좌표 적용 완료 · 테두리와 주요 구조가 비슷하게 겹치는지 확인하세요.";
    showToast("통합 XY 좌표 이미지를 만들었습니다.");
  } catch (error) {
    alignmentMessage.classList.add("error");
    alignmentMessage.textContent = error.message;
  } finally {
    alignmentButton.querySelector("b").textContent = "→";
    updateCompletion();
  }
}

function payloadFor(cameraId) {
  const camera = state.cameras[cameraId];
  return {
    camera_id: cameraId,
    image_id: camera.imageId,
    points: camera.points.map(point => ({ x: round(point.x), y: round(point.y) })),
    length_m: Number(lengthInput.value),
    width_m: Number(widthInput.value),
    rotation_degrees: camera.rotationDegrees,
    flip_horizontal: camera.flipHorizontal,
  };
}

async function saveConfig() {
  saveButton.disabled = true;
  saveMessage.textContent = "설정을 확인하고 있습니다…";
  try {
    const response = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pool: { length_m: Number(lengthInput.value), width_m: Number(widthInput.value) },
        cameras: [payloadFor("camera1"), payloadFor("camera2")],
      }),
    });
    const data = await readResponse(response);
    const savedColorCalibration = {
      ...data.config.color_calibration,
      cameras: Object.fromEntries(data.config.cameras.map(camera => [camera.id, camera.image_adjustments])),
    };
    state.colorCalibration = savedColorCalibration;
    renderColorCalibration(savedColorCalibration);
    saveMessage.textContent = data.message;
    document.querySelector("#download-config").hidden = false;
    showToast("설정 저장 완료");
  } catch (error) {
    saveMessage.textContent = error.message;
  } finally {
    updateCompletion();
  }
}

function renderColorCalibration(calibration) {
  const empty = document.querySelector("#color-empty");
  const results = document.querySelector("#color-results");
  if (!empty || !results) return;
  if (!calibration?.cameras) {
    empty.hidden = false;
    results.hidden = true;
    return;
  }

  for (const cameraId of ["camera1", "camera2"]) {
    const values = calibration.cameras[cameraId];
    const card = results.querySelector(`[data-color-camera="${cameraId}"]`);
    card.querySelector(".wb-red").textContent = `R ×${values.white_balance.red_gain_multiplier.toFixed(3)}`;
    card.querySelector(".wb-blue").textContent = `B ×${values.white_balance.blue_gain_multiplier.toFixed(3)}`;
    const ev = values.exposure.compensation_ev;
    card.querySelector(".exposure-ev").textContent = `${ev >= 0 ? "+" : ""}${ev.toFixed(2)} EV`;
  }
  empty.hidden = true;
  results.hidden = false;
}

function updateCompletion() {
  let allDone = !validateDimensions();
  const orientedSizes = [];
  for (const cameraId of ["camera1", "camera2"]) {
    const done = state.cameras[cameraId].calibrated;
    document.querySelector(`#done-${cameraId}`).classList.toggle("done", done);
    allDone = allDone && done;
    if (done) orientedSizes.push(JSON.stringify(state.cameras[cameraId].calibration.oriented_coordinate_size_m));
  }
  const sameDirection = orientedSizes.length === 2 && orientedSizes[0] === orientedSizes[1];
  alignmentButton.disabled = !allDone || !sameDirection;
  if (allDone && !sameDirection) {
    alignmentMessage.classList.add("error");
    alignmentMessage.textContent = "두 미리보기의 가로·세로 방향이 다릅니다. 한쪽을 90° 돌려 주세요.";
  } else if (!allDone && !state.alignmentReady) {
    alignmentMessage.classList.remove("error");
    alignmentMessage.textContent = "두 카메라 미리보기를 먼저 완료해 주세요.";
  }
  saveButton.disabled = !allDone || !state.alignmentReady;
}

function setCardState(card, label, ready = false) {
  const badge = card.querySelector(".camera-state");
  badge.textContent = label;
  badge.classList.toggle("ready", ready);
}

function showError(card, message) { card.querySelector(".camera-error").textContent = message; }
function clearError(card) { showError(card, ""); }
function clamp(value, minimum, maximum) { return Math.max(minimum, Math.min(maximum, value)); }
function round(value) { return Math.round(value * 1000) / 1000; }

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "요청을 처리하지 못했습니다.");
  return data;
}

let toastTimer;
function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2600);
}

window.addEventListener("resize", () => {
  for (const cameraId of ["camera1", "camera2"]) {
    if (state.cameras[cameraId].image) draw(cameraId, cameraList.querySelector(`[data-camera-id="${cameraId}"]`));
  }
});
