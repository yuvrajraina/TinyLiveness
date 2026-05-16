import "./styles.css";

const IMAGE_SIZE = 224;
const DEFAULT_API_URL =
  import.meta.env.VITE_TINYLIVENESS_API_URL ||
  "http://127.0.0.1:8000/api/liveness/predict/";

const tester = {
  modelStatus: document.querySelector("#modelStatus"),
  servedModelName: document.querySelector("#servedModelName"),
  servedModelStatus: document.querySelector("#servedModelStatus"),
  servedModelPolicy: document.querySelector("#servedModelPolicy"),
  servedModelThreshold: document.querySelector("#servedModelThreshold"),
  video: document.querySelector("#videoPreview"),
  canvas: document.querySelector("#inputCanvas"),
  imageInput: document.querySelector("#imageInput"),
  startCamera: document.querySelector("#startCamera"),
  captureFrame: document.querySelector("#captureFrame"),
  runCheck: document.querySelector("#runCheck"),
  scoreValue: document.querySelector("#scoreValue"),
  scoreReadout: document.querySelector("#scoreReadout"),
  spoofReadout: document.querySelector("#spoofReadout"),
  scoreBar: document.querySelector("#scoreBar"),
  decisionValue: document.querySelector("#decisionValue"),
  thresholdValue: document.querySelector("#thresholdValue"),
  processingMs: document.querySelector("#processingMs"),
  rawResult: document.querySelector("#rawResult")
};

let stream = null;
let activeImageData = null;
let context = null;

markActiveNav();

if (tester.canvas) {
  context = tester.canvas.getContext("2d", { willReadFrequently: true });
  drawPlaceholder();
  bindTester();
  loadModelInfo();
}

function markActiveNav() {
  const currentPage = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".site-header nav a").forEach((link) => {
    const href = link.getAttribute("href") || "";
    const hrefPage = href.split("/").pop() || "index.html";
    link.classList.toggle("active", hrefPage === currentPage);
  });
}

function bindTester() {
  tester.imageInput.addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) {
      return;
    }
    const image = await loadImage(file);
    drawSourceToCanvas(image);
    tester.video.classList.remove("is-visible");
    tester.runCheck.disabled = false;
    setStatus("Image ready");
  });

  tester.startCamera.addEventListener("click", async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      tester.video.srcObject = stream;
      await tester.video.play();
      tester.video.classList.add("is-visible");
      tester.captureFrame.disabled = false;
      setStatus("Camera ready");
    } catch (error) {
      setStatus(`Camera unavailable: ${error.message}`, true);
    }
  });

  tester.captureFrame.addEventListener("click", () => {
    if (!tester.video.videoWidth) {
      setStatus("Camera frame is not ready yet.", true);
      return;
    }
    drawSourceToCanvas(tester.video);
    tester.video.classList.remove("is-visible");
    tester.runCheck.disabled = false;
    setStatus("Frame captured");
  });

  tester.runCheck.addEventListener("click", async () => {
    if (!activeImageData) {
      setStatus("Add an image or capture a frame first.", true);
      return;
    }

    tester.runCheck.disabled = true;
    try {
      setStatus("Calling liveness API");
      const result = await callBackend();
      renderResult(result);
    } catch (error) {
      setStatus(error.message || "Liveness check failed.", true);
      renderError(error);
    } finally {
      tester.runCheck.disabled = false;
    }
  });

}

function setStatus(message, isError = false) {
  tester.modelStatus.textContent = message;
  tester.modelStatus.classList.toggle("is-error", isError);
}

function drawPlaceholder() {
  const width = tester.canvas.width;
  context.fillStyle = "#eef2f8";
  context.fillRect(0, 0, width, width);
  context.strokeStyle = "#175cd3";
  context.lineWidth = 2;
  context.strokeRect(28, 28, 168, 168);
  context.beginPath();
  context.arc(112, 94, 36, 0, Math.PI * 2);
  context.stroke();
  context.beginPath();
  context.ellipse(112, 158, 58, 38, 0, 0, Math.PI * 2);
  context.stroke();
}

function drawSourceToCanvas(source) {
  const sourceWidth = source.videoWidth || source.naturalWidth || source.width;
  const sourceHeight = source.videoHeight || source.naturalHeight || source.height;
  const side = Math.min(sourceWidth, sourceHeight);
  const sx = (sourceWidth - side) / 2;
  const sy = (sourceHeight - side) / 2;

  context.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  context.drawImage(source, sx, sy, side, side, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  activeImageData = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not load that image."));
    };
    image.src = url;
  });
}

async function canvasBlob() {
  return new Promise((resolve, reject) => {
    tester.canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Could not encode the frame."));
        }
      },
      "image/jpeg",
      0.92
    );
  });
}

function modelInfoUrl() {
  const endpoint = DEFAULT_API_URL;
  const url = new URL(endpoint);
  url.pathname = url.pathname.replace(/\/predict\/?$/, "/model/");
  return url.toString();
}

async function loadModelInfo() {
  if (!tester.servedModelName) {
    return;
  }
  tester.servedModelName.textContent = "Checking API model...";
  tester.servedModelStatus.textContent = "Loading";
  tester.servedModelPolicy.textContent = "-";
  tester.servedModelThreshold.textContent = "-";

  try {
    const response = await fetch(modelInfoUrl());
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || data.detail || "model unavailable");
    }
    tester.servedModelName.textContent = data.model_name || "TinyLiveness model";
    tester.servedModelStatus.textContent = data.status || "ready";
    tester.servedModelPolicy.textContent = data.threshold_policy || data.model_variant || "-";
    tester.servedModelThreshold.textContent =
      typeof data.accept_threshold === "number" ? data.accept_threshold.toFixed(6) : "-";
  } catch (error) {
    tester.servedModelName.textContent = "API model unavailable";
    tester.servedModelStatus.textContent = error.message;
  }
}

async function callBackend() {
  const endpoint = DEFAULT_API_URL;

  const formData = new FormData();
  formData.append("image", await canvasBlob(), "tinyliveness-frame.jpg");

  const response = await fetch(endpoint, { method: "POST", body: formData });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.detail || payload?.error || `API returned ${response.status}`);
  }
  return payload;
}

function renderResult(result) {
  const probability = Number(result.live_probability);
  const spoofProbability = Number(result.spoof_probability);
  const decision = result.decision || "unknown";
  const acceptThreshold = Number(result.accept_threshold ?? result.threshold);
  const rejectThreshold = Number(result.reject_threshold ?? acceptThreshold);
  const thresholdPolicy = result.threshold_policy || "unknown_policy";

  tester.scoreValue.textContent = `${(probability * 100).toFixed(2)}%`;
  tester.scoreReadout.textContent = `${(probability * 100).toFixed(2)}%`;
  tester.spoofReadout.textContent = `${(spoofProbability * 100).toFixed(2)}%`;
  tester.scoreBar.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
  tester.decisionValue.textContent = decision.replace("_", " ");
  tester.decisionValue.dataset.decision = decision;
  tester.thresholdValue.textContent =
    `reject ${rejectThreshold.toFixed(6)} / accept ${acceptThreshold.toFixed(6)} / ${thresholdPolicy}`;
  tester.processingMs.textContent =
    typeof result.processing_ms === "number" ? `${Math.round(result.processing_ms)} ms` : "-";
  tester.rawResult.textContent = JSON.stringify(result, null, 2);
  setStatus("API result ready");
}

function renderError(error) {
  tester.decisionValue.textContent = "Error";
  tester.decisionValue.dataset.decision = "spoof";
  tester.rawResult.textContent = JSON.stringify(
    {
      error: "request_failed",
      detail: error.message
    },
    null,
    2
  );
}
