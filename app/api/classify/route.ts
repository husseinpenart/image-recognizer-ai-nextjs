import { type NextRequest, NextResponse } from "next/server";
import * as ort from "onnxruntime-node";
import sharp from "sharp";
import fs from "fs";
import path from "path";
import { translateToFarsi } from "@/app/utils/translateToFarsi";

// COCO labels
const COCO_LABELS = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

interface Detection {
  label: string;
  confidence: number;
  box: number[];
}

// Preprocess image for YOLOv5
function preprocessImage(imageBuffer: Buffer): Float32Array {
  const pixels = new Float32Array(640 * 640 * 3);

  // Normalize to [0, 1] and convert to NCHW format
  for (let i = 0; i < imageBuffer.length; i += 3) {
    const pixelIndex = Math.floor(i / 3);
    const y = Math.floor(pixelIndex / 640);
    const x = pixelIndex % 640;

    // RGB to NCHW: [R_channel, G_channel, B_channel]
    pixels[y * 640 + x] = imageBuffer[i] / 255.0; // R channel
    pixels[640 * 640 + y * 640 + x] = imageBuffer[i + 1] / 255.0; // G channel
    pixels[2 * 640 * 640 + y * 640 + x] = imageBuffer[i + 2] / 255.0; // B channel
  }

  return pixels;
}

// Convert Float16 (Uint16Array) to Float32Array
function float16ToFloat32(float16Array: Uint16Array): Float32Array {
  const float32Array = new Float32Array(float16Array.length);

  for (let i = 0; i < float16Array.length; i++) {
    const h = float16Array[i];

    const sign = (h & 0x8000) >> 15;
    const exponent = (h & 0x7c00) >> 10;
    const mantissa = h & 0x03ff;

    let value: number;

    if (exponent === 0) {
      if (mantissa === 0) {
        // Zero
        value = 0;
      } else {
        // Subnormal
        value = Math.pow(-1, sign) * Math.pow(2, -14) * (mantissa / 1024);
      }
    } else if (exponent === 31) {
      if (mantissa === 0) {
        // Infinity
        value = sign ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
      } else {
        // NaN
        value = Number.NaN;
      }
    } else {
      // Normal
      value =
        Math.pow(-1, sign) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
    }

    float32Array[i] = value;
  }

  return float32Array;
}

// Sigmoid function
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// Process YOLOv5 output - corrected version
function processYoloOutput(
  output: Float32Array,
  originalWidth: number,
  originalHeight: number,
  confThreshold = 0.25,
  iouThreshold = 0.45
): Detection[] {
  console.log("Processing YOLO output...");
  console.log("Output length:", output.length);

  // YOLOv5 output format: [1, 25200, 85]
  // 25200 = number of detections
  // 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
  const numDetections = output.length / 85;
  console.log("Estimated number of detections:", numDetections);

  const detections: Detection[] = [];

  // Debug first few outputs to understand the format
  console.log(
    "First detection data (converted to float32):",
    output.slice(0, 85)
  );

  for (let i = 0; i < numDetections; i++) {
    const baseIdx = i * 85;

    // Extract bbox coordinates (center_x, center_y, width, height)
    const centerX = output[baseIdx];
    const centerY = output[baseIdx + 1];
    const width = output[baseIdx + 2];
    const height = output[baseIdx + 3];

    // Extract objectness confidence and apply sigmoid
    const objectness = sigmoid(output[baseIdx + 4]);

    if (objectness < confThreshold) {
      continue;
    }

    // Find the class with highest probability
    let maxClassProb = 0;
    let maxClassIndex = 0;

    for (let j = 0; j < 80; j++) {
      const classProb = sigmoid(output[baseIdx + 5 + j]);
      if (classProb > maxClassProb) {
        maxClassProb = classProb;
        maxClassIndex = j;
      }
    }

    // Calculate final confidence
    const confidence = objectness * maxClassProb;

    if (confidence < confThreshold) {
      continue;
    }

    // Convert from center format to corner format and scale to original image size
    const x1 = Math.max(0, ((centerX - width / 2) / 640) * originalWidth);
    const y1 = Math.max(0, ((centerY - height / 2) / 640) * originalHeight);
    const x2 = Math.min(
      originalWidth,
      ((centerX + width / 2) / 640) * originalWidth
    );
    const y2 = Math.min(
      originalHeight,
      ((centerY + height / 2) / 640) * originalHeight
    );

    // Only add if the box has reasonable dimensions
    if (x2 > x1 && y2 > y1) {
      detections.push({
        label: COCO_LABELS[maxClassIndex] || "unknown",
        confidence: confidence,
        box: [x1, y1, x2, y2],
      });

      // Log first few detections for debugging
      if (detections.length <= 5) {
        console.log(`Detection ${detections.length}:`, {
          label: COCO_LABELS[maxClassIndex],
          confidence: confidence,
          objectness: objectness,
          maxClassProb: maxClassProb,
          centerX,
          centerY,
          width,
          height,
          box: [x1, y1, x2, y2],
        });
      }
    }
  }

  console.log(`Found ${detections.length} raw detections`);

  // Apply Non-Maximum Suppression
  const finalDetections = applyNMS(detections, iouThreshold);
  console.log(`After NMS: ${finalDetections.length} detections`);

  return finalDetections;
}

// Non-Maximum Suppression
function applyNMS(detections: Detection[], iouThreshold: number): Detection[] {
  // Sort by confidence (highest first)
  const sorted = detections.sort((a, b) => b.confidence - a.confidence);
  const selected: Detection[] = [];

  for (const detection of sorted) {
    let keep = true;

    for (const selectedDetection of selected) {
      if (calculateIoU(detection.box, selectedDetection.box) > iouThreshold) {
        keep = false;
        break;
      }
    }

    if (keep) {
      selected.push(detection);
    }
  }

  return selected;
}

// Calculate Intersection over Union
function calculateIoU(box1: number[], box2: number[]): number {
  const [x1_1, y1_1, x2_1, y2_1] = box1;
  const [x1_2, y1_2, x2_2, y2_2] = box2;

  // Calculate intersection area
  const x1 = Math.max(x1_1, x1_2);
  const y1 = Math.max(y1_1, y1_2);
  const x2 = Math.min(x2_1, x2_2);
  const y2 = Math.min(y2_1, y2_2);

  const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

  // Calculate union area
  const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
  const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
  const unionArea = area1 + area2 - intersectionArea;

  return intersectionArea / unionArea;
}

// Persian translations

// Convert Float32Array to Float16 for model input
function float32ToFloat16(float32Array: Float32Array): Uint16Array {
  const float16Array = new Uint16Array(float32Array.length);

  for (let i = 0; i < float32Array.length; i++) {
    const value = float32Array[i];

    // Handle special cases
    if (value === 0) {
      float16Array[i] = 0;
      continue;
    }

    if (!isFinite(value)) {
      float16Array[i] = value > 0 ? 0x7c00 : 0xfc00; // +inf or -inf
      continue;
    }

    // Convert to float16
    const float32 = new Float32Array([value]);
    const int32 = new Int32Array(float32.buffer)[0];

    const sign = (int32 >>> 31) << 15;
    let exponent = ((int32 >>> 23) & 0xff) - 127 + 15;
    let mantissa = int32 & 0x7fffff;

    if (exponent <= 0) {
      // Subnormal or zero
      mantissa = mantissa | 0x800000;
      mantissa = mantissa >>> (1 - exponent);
      exponent = 0;
    } else if (exponent >= 31) {
      // Infinity or NaN
      exponent = 31;
      mantissa = mantissa ? 0x200 : 0;
    } else {
      mantissa = mantissa >>> 13;
    }

    float16Array[i] = sign | (exponent << 10) | mantissa;
  }

  return float16Array;
}

export async function POST(req: NextRequest) {
  try {
    const { filename } = await req.json();
    const imagePath = path.join(process.cwd(), "public/uploads", filename);

    if (!fs.existsSync(imagePath)) {
      return NextResponse.json({ error: "تصویر یافت نشد" }, { status: 404 });
    }

    // Get original image dimensions
    const metadata = await sharp(imagePath).metadata();
    const originalWidth = metadata.width!;
    const originalHeight = metadata.height!;

    console.log(
      "Original image dimensions:",
      originalWidth,
      "x",
      originalHeight
    );

    // Preprocess image
    const imageBuffer = await sharp(imagePath)
      .resize(640, 640)
      .removeAlpha() // Remove alpha channel if present
      .raw()
      .toBuffer();

    console.log("Preprocessed image buffer length:", imageBuffer.length);

    const preprocessedData = preprocessImage(imageBuffer);
    const float16Data = float32ToFloat16(preprocessedData);

    // Create input tensor with float16
    const inputTensor = new ort.Tensor(
      "float16",
      float16Data,
      [1, 3, 640, 640]
    );

    // Load and run model
    const modelPath = path.join(process.cwd(), "models", "yolov5l.onnx");
    if (!fs.existsSync(modelPath)) {
      return NextResponse.json({ error: "مدل یافت نشد" }, { status: 404 });
    }

    console.log("Loading model...");
    const session = await ort.InferenceSession.create(modelPath);
    console.log("Model input names:", session.inputNames);
    console.log("Model output names:", session.outputNames);

    // Run inference
    const feeds: Record<string, ort.Tensor> = {
      [session.inputNames[0]]: inputTensor,
    };

    console.log("Running inference...");
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];

    console.log("Model output shape:", output.dims);
    console.log("Output data type:", typeof output.data);
    console.log("Output data constructor:", output.data.constructor.name);

    // Convert output data to Float32Array if it's Uint16Array (float16)
    let outputData: Float32Array;
    if (output.data instanceof Uint16Array) {
      console.log("Converting float16 output to float32...");
      outputData = float16ToFloat32(output.data);
    } else {
      outputData = output.data as Float32Array;
    }

    console.log("Output data length:", outputData.length);

    // Process detections with a higher confidence threshold to reduce false positives
    const detections = processYoloOutput(
      outputData,
      originalWidth,
      originalHeight,
      0.25, // Higher confidence threshold
      0.45 // IoU threshold
    );

    // Format results with proper confidence percentage
    const result = detections.map((det) => ({
      label: det.label,
      label_fa: translateToFarsi(det.label),
      confidence: `${(det.confidence * 100).toFixed(2)}%`,
      box: det.box.map((coord) => Math.round(coord)),
    }));

    console.log("Final detections:", result.length);
    return NextResponse.json(result);
  } catch (error) {
    console.error("Error in image detection:", error);
    return NextResponse.json({ error: "خطای سرور" }, { status: 500 });
  }
}
