import * as ort from "onnxruntime-web";

const DEFAULT_IMAGE_SIZE = 224;
const CHANNELS = 3;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

export function configureOnnxRuntimeWeb(options = {}) {
  if (options.wasmPaths) {
    ort.env.wasm.wasmPaths = options.wasmPaths;
  }
  if (options.numThreads !== undefined) {
    ort.env.wasm.numThreads = options.numThreads;
  }
  if (options.simd !== undefined) {
    ort.env.wasm.simd = options.simd;
  }
}

function resolveThresholdPolicy(options = {}, thresholdsJson = null) {
  let rejectThreshold = options.rejectThreshold ?? options.threshold ?? 0.5;
  let acceptThreshold = options.acceptThreshold ?? options.threshold ?? 0.5;
  let thresholdPolicy = options.thresholdPolicy ?? "single_threshold";

  const payload = thresholdsJson;
  if (payload?.decision_policy) {
    rejectThreshold = payload.decision_policy.reject_threshold ?? rejectThreshold;
    acceptThreshold = payload.decision_policy.accept_threshold ?? payload.threshold ?? acceptThreshold;
    thresholdPolicy = payload.decision_policy.threshold_policy ?? payload.policy ?? thresholdPolicy;
  } else if (payload?.production) {
    acceptThreshold = payload.production.threshold;
    rejectThreshold = acceptThreshold;
    thresholdPolicy = payload.production.policy ?? "production_threshold";
  } else if (payload?.threshold !== undefined) {
    acceptThreshold = payload.threshold;
    rejectThreshold = acceptThreshold;
    thresholdPolicy = payload.policy ?? "threshold_json";
  }

  if (rejectThreshold > acceptThreshold) {
    throw new Error("rejectThreshold must be <= acceptThreshold");
  }
  return { rejectThreshold, acceptThreshold, thresholdPolicy };
}

function decisionFromProbability(probability, rejectThreshold, acceptThreshold) {
  if (probability < rejectThreshold) {
    return "spoof";
  }
  if (probability >= acceptThreshold) {
    return "live";
  }
  return "manual_review";
}

function sigmoid(value) {
  return 1.0 / (1.0 + Math.exp(-value));
}

function applyScoreCalibration(probability, calibration) {
  if (!calibration) {
    return probability;
  }
  const isotonic = calibration.isotonic?.model;
  if (isotonic?.x_thresholds?.length && isotonic?.y_thresholds?.length) {
    const x = isotonic.x_thresholds;
    const y = isotonic.y_thresholds;
    if (probability <= x[0]) {
      return y[0];
    }
    for (let index = 1; index < x.length; index += 1) {
      if (probability <= x[index]) {
        const span = Math.max(x[index] - x[index - 1], 1e-12);
        const ratio = (probability - x[index - 1]) / span;
        return y[index - 1] + ratio * (y[index] - y[index - 1]);
      }
    }
    return y[y.length - 1];
  }
  const temperature = calibration.temperature_scaling?.temperature;
  if (temperature) {
    const clipped = Math.min(1.0 - 1e-6, Math.max(1e-6, probability));
    const logit = Math.log(clipped / (1.0 - clipped));
    return sigmoid(logit / Math.max(temperature, 1e-6));
  }
  return probability;
}

export function normalizeRgbPixels(
  rgbPixels,
  width = DEFAULT_IMAGE_SIZE,
  height = DEFAULT_IMAGE_SIZE,
  normalization = "tinyliveness"
) {
  if (width !== height) {
    throw new Error("expected a square RGB crop");
  }
  const imageSize = width;
  if (rgbPixels.length !== imageSize * imageSize * CHANNELS) {
    throw new Error(`expected RGB pixel array shaped ${imageSize}x${imageSize}x3`);
  }

  const output = new Float32Array(CHANNELS * imageSize * imageSize);
  const plane = imageSize * imageSize;
  for (let index = 0; index < plane; index += 1) {
    const source = index * CHANNELS;
    if (normalization === "tinyliveness") {
      output[index] = (rgbPixels[source] - 127.5) / 128.0;
      output[plane + index] = (rgbPixels[source + 1] - 127.5) / 128.0;
      output[plane * 2 + index] = (rgbPixels[source + 2] - 127.5) / 128.0;
    } else if (normalization === "imagenet") {
      output[index] = (rgbPixels[source] / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      output[plane + index] = (rgbPixels[source + 1] / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      output[plane * 2 + index] = (rgbPixels[source + 2] / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    } else {
      throw new Error(`unknown normalization: ${normalization}`);
    }
  }
  return output;
}

export function imageDataToRgbPixels(imageData, imageSize = DEFAULT_IMAGE_SIZE) {
  if (imageData.width !== imageSize || imageData.height !== imageSize) {
    throw new Error(`expected ${imageSize}x${imageSize} ImageData`);
  }
  const rgb = new Uint8Array(imageSize * imageSize * CHANNELS);
  for (let pixel = 0; pixel < imageSize * imageSize; pixel += 1) {
    const rgbaIndex = pixel * 4;
    const rgbIndex = pixel * 3;
    rgb[rgbIndex] = imageData.data[rgbaIndex];
    rgb[rgbIndex + 1] = imageData.data[rgbaIndex + 1];
    rgb[rgbIndex + 2] = imageData.data[rgbaIndex + 2];
  }
  return rgb;
}

export function aggregateScores(scores, method = "mean") {
  if (!scores.length) {
    throw new Error("at least one frame score is required");
  }
  const sorted = [...scores].sort((left, right) => left - right);
  if (method === "mean") {
    return scores.reduce((sum, value) => sum + value, 0) / scores.length;
  }
  if (method === "median") {
    return sorted[Math.floor(sorted.length / 2)];
  }
  if (method === "min") {
    return sorted[0];
  }
  if (method === "p10") {
    return sorted[Math.floor((sorted.length - 1) * 0.10)];
  }
  throw new Error(`unknown aggregation method: ${method}`);
}

export class TinyLivenessSession {
  constructor(
    session,
    thresholdPolicy = resolveThresholdPolicy(),
    normalization = "tinyliveness",
    imageSize = DEFAULT_IMAGE_SIZE,
    scoreCalibration = null
  ) {
    this.session = session;
    if (typeof thresholdPolicy === "number") {
      thresholdPolicy = resolveThresholdPolicy({ threshold: thresholdPolicy });
    }
    this.rejectThreshold = thresholdPolicy.rejectThreshold;
    this.acceptThreshold = thresholdPolicy.acceptThreshold;
    this.threshold = this.acceptThreshold;
    this.thresholdPolicy = thresholdPolicy.thresholdPolicy;
    this.normalization = normalization;
    this.imageSize = imageSize;
    this.scoreCalibration = scoreCalibration;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
  }

  static async create(modelUrl, options = {}) {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: options.executionProviders || ["wasm"],
      graphOptimizationLevel: "all"
    });
    let thresholdsJson = options.thresholdsJson ?? null;
    if (!thresholdsJson && options.thresholdsUrl) {
      thresholdsJson = await (await fetch(options.thresholdsUrl)).json();
    }
    let scoreCalibration = options.scoreCalibration ?? null;
    if (!scoreCalibration && options.scoreCalibrationUrl) {
      scoreCalibration = await (await fetch(options.scoreCalibrationUrl)).json();
    }
    return new TinyLivenessSession(
      session,
      resolveThresholdPolicy(options, thresholdsJson),
      options.normalization || "tinyliveness",
      options.imageSize || DEFAULT_IMAGE_SIZE,
      scoreCalibration
    );
  }

  async predictTensor(chwTensor) {
    const input = new ort.Tensor("float32", chwTensor, [1, 3, this.imageSize, this.imageSize]);
    const outputs = await this.session.run({ [this.inputName]: input });
    const probability = applyScoreCalibration(Number(outputs[this.outputName].data[0]), this.scoreCalibration);
    const decision = decisionFromProbability(probability, this.rejectThreshold, this.acceptThreshold);
    return {
      liveProbability: probability,
      spoofProbability: 1.0 - probability,
      isLive: decision === "live",
      isSpoof: decision === "spoof",
      decision,
      threshold: this.acceptThreshold,
      rejectThreshold: this.rejectThreshold,
      acceptThreshold: this.acceptThreshold,
      thresholdPolicy: this.thresholdPolicy
    };
  }

  async predictImageData(imageData) {
    return this.predictTensor(
      normalizeRgbPixels(
        imageDataToRgbPixels(imageData, this.imageSize),
        this.imageSize,
        this.imageSize,
        this.normalization
      )
    );
  }

  async predictSequence(imageDataFrames, aggregation = "mean") {
    const frameResults = [];
    for (const imageData of imageDataFrames) {
      frameResults.push(await this.predictImageData(imageData));
    }
    const frameProbabilities = frameResults.map((result) => result.liveProbability);
    const liveProbability = aggregateScores(frameProbabilities, aggregation);
    const decision = decisionFromProbability(liveProbability, this.rejectThreshold, this.acceptThreshold);
    return {
      liveProbability,
      spoofProbability: 1.0 - liveProbability,
      isLive: decision === "live",
      isSpoof: decision === "spoof",
      decision,
      threshold: this.acceptThreshold,
      rejectThreshold: this.rejectThreshold,
      acceptThreshold: this.acceptThreshold,
      thresholdPolicy: this.thresholdPolicy,
      aggregation,
      frameProbabilities
    };
  }
}
