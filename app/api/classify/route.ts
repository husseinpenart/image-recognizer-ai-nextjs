import { NextRequest, NextResponse } from "next/server";
import * as ort from "onnxruntime-node";
import sharp from "sharp";
import fs from "fs";
import path from "path";

// مسیر فایل لیبل‌ها
const labelsPath = path.join(process.cwd(), "models", "coco_labels.json");
let labels: string[] = [];
try {
  console.log("Reading labels from:", labelsPath);
  labels = JSON.parse(fs.readFileSync(labelsPath, "utf-8"));
  console.log("Loaded labels:", labels.slice(0, 5), "...", labels.slice(-5)); // چک کردن لیبل‌ها
} catch (error) {
  console.error("Error reading labels:", error);
}

// تابع نرمال‌سازی تصویر برای YOLOv5 (اصلاح‌شده)
function preprocessImage(data: Buffer): Float32Array {
  console.log("Raw image buffer length:", data.length); // دیباگ طول داده خام
  if (data.length !== 640 * 640 * 3) {
    console.error("Unexpected image buffer length:", data.length);
    throw new Error("Unexpected image buffer length");
  }

  // نرمال‌سازی داده‌های خام به [0, 1]
  const pixels = new Float32Array(640 * 640 * 3);
  for (let i = 0; i < data.length; i++) {
    pixels[i] = data[i] / 255.0; // نرمال‌سازی به [0, 1]
  }
  console.log("Sample raw pixels:", pixels.slice(0, 15)); // دیباگ مقادیر خام پیکسل‌ها (15 تا برای دیدن RGB)

  // تبدیل به فرمت NCHW (1, 3, 640, 640)
  const nchw = new Float32Array(1 * 3 * 640 * 640);
  for (let y = 0; y < 640; y++) {
    for (let x = 0; x < 640; x++) {
      const idx = (y * 640 + x) * 3;
      const r = pixels[idx];
      const g = pixels[idx + 1];
      const b = pixels[idx + 2];
      nchw[0 * 640 * 640 + y * 640 + x] = r; // R
      nchw[1 * 640 * 640 + y * 640 + x] = g; // G
      nchw[2 * 640 * 640 + y * 640 + x] = b; // B
    }
  }
  console.log("Sample preprocessed data:", nchw.slice(0, 15)); // دیباگ داده خروجی
  return nchw;
}

// تبدیل Float32Array به Float16Array
function toFloat16Array(float32Array: Float32Array): Uint16Array {
  const float16Array = new Uint16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    float16Array[i] = float32ToFloat16(float32Array[i]);
  }
  return float16Array;
}

// تابع کمکی برای تبدیل float32 به float16
function float32ToFloat16(value: number): number {
  const float = new Float32Array([value]);
  const bytes = new Uint32Array(float.buffer);
  const bits = bytes[0];

  const sign = bits & 0x80000000;
  let exponent = (bits & 0x7f800000) >> 23;
  const mantissa = bits & 0x007fffff;

  if (exponent === 0xff) {
    exponent = 0x1f;
  } else if (exponent === 0) {
    exponent = 0;
  } else {
    exponent = exponent - 127 + 15;
    if (exponent > 0x1f) exponent = 0x1f;
    else if (exponent <= 0) exponent = 0;
  }

  const mantissa16 = mantissa >> 13;
  return (sign >> 16) | (exponent << 10) | mantissa16;
}

// تابع پردازش خروجی YOLOv5
function processYoloOutput(
  output: Float32Array,
  imgWidth: number,
  imgHeight: number,
  confThreshold = 0.5,
  iouThreshold = 0.4
) {
  const detections: Array<{
    label: string;
    confidence: number;
    box: number[];
  }> = [];
  const strides = [8, 16, 32]; // مقیاس‌های YOLOv5
  const anchors = [
    [10, 13, 16, 30, 33, 23], // برای stride 8
    [30, 61, 62, 45, 59, 119], // برای stride 16
    [116, 90, 156, 198, 373, 326], // برای stride 32
  ];
  const numAnchors = 3; // هر مقیاس 3 anchor داره

  console.log("Output shape:", output.length, "Dims:", output.dims); // دیباگ شکل خروجی
  console.log("Sample output data:", output.slice(0, 15)); // دیباگ مقادیر خام

  let offset = 0;
  for (let s = 0; s < strides.length; s++) {
    const stride = strides[s];
    const gridSize = (640 / stride) * (640 / stride);
    const anchorGroup = anchors[s];

    for (let a = 0; a < numAnchors; a++) {
      for (let i = 0; i < gridSize; i++) {
        const idx = offset + (i * numAnchors + a) * (5 + 80); // 5: [x, y, w, h, conf], 80: classes
        if (idx + 4 >= output.length) {
          console.log(
            "Index out of bounds at stride",
            stride,
            "anchor",
            a,
            "grid",
            i
          );
          continue;
        }

        const confidence = 1 / (1 + Math.exp(-output[idx + 4])); // sigmoid on confidence
        if (confidence < confThreshold) continue;

        // پیدا کردن کلاس با بالاترین احتمال
        let maxClassScore = -Infinity;
        let maxClassIndex = 0;
        for (let j = 0; j < 80; j++) {
          const score = 1 / (1 + Math.exp(-output[idx + 5 + j])); // sigmoid on class scores
          if (score > maxClassScore) {
            maxClassScore = score;
            maxClassIndex = j;
          }
        }

        const finalConfidence = confidence * maxClassScore;
        if (finalConfidence < confThreshold) continue;

        // محاسبه مختصات کادر
        const x = (output[idx] * 2 - 0.5 + (i % (640 / stride))) * stride;
        const y =
          (output[idx + 1] * 2 - 0.5 + Math.floor(i / (640 / stride))) * stride;
        const w = Math.exp(output[idx + 2]) * anchorGroup[a * 2];
        const h = Math.exp(output[idx + 3]) * anchorGroup[a * 2 + 1];
        const box = [
          ((x - w / 2) / 640) * imgWidth,
          ((y - h / 2) / 640) * imgHeight,
          ((x + w / 2) / 640) * imgWidth,
          ((y + h / 2) / 640) * imgHeight,
        ];

        // console.log(
        //   `Detection (stride ${stride}, anchor ${a}, grid ${i}): Confidence: ${confidence.toFixed(
        //     4
        //   )}, Max class score: ${maxClassScore.toFixed(
        //     4
        //   )}, Class index: ${maxClassIndex}, Label: ${labels[maxClassIndex]}`
        // );

        detections.push({
          label: labels[maxClassIndex] || "unknown",
          confidence: finalConfidence,
          box,
        });
      }
    }
    offset += gridSize * numAnchors * (5 + 80);
  }

  return applyNMS(detections, iouThreshold);
}

// تابع NMS (Non-Maximum Suppression)
function applyNMS(
  detections: Array<{ label: string; confidence: number; box: number[] }>,
  iouThreshold: number
) {
  const sorted = detections.sort((a, b) => b.confidence - a.confidence);
  const selected: typeof detections = [];

  for (const det of sorted) {
    let keep = true;
    for (const sel of selected) {
      if (calculateIoU(det.box, sel.box) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) selected.push(det);
  }

  return selected;
}

// محاسبه IoU (Intersection over Union)
function calculateIoU(box1: number[], box2: number[]): number {
  const [x1_min, y1_min, x1_max, y1_max] = box1;
  const [x2_min, y2_min, x2_max, y2_max] = box2;

  const x_min = Math.max(x1_min, x2_min);
  const y_min = Math.max(y1_min, y2_min);
  const x_max = Math.min(x1_max, x2_max);
  const y_max = Math.min(y1_max, y2_max);

  const intersection = Math.max(0, x_max - x_min) * Math.max(0, y_max - y_min);
  const area1 = (x1_max - x1_min) * (y1_max - y1_min);
  const area2 = (x2_max - x2_min) * (y2_max - y2_min);
  return intersection / (area1 + area2 - intersection);
}

export async function POST(req: NextRequest) {
  try {
    const { filename } = await req.json();
    const imagePath = path.join(process.cwd(), "public/uploads", filename);

    if (!fs.existsSync(imagePath)) {
      return NextResponse.json({ error: "تصویر یافت نشد" }, { status: 404 });
    }

    // خواندن ابعاد اصلی تصویر
    const { width, height } = await sharp(imagePath).metadata();
    console.log("Image dimensions:", width, "x", height); // دیباگ ابعاد تصویر

    // پیش‌پردازش تصویر
    const imageBuffer = await sharp(imagePath)
      .resize(640, 640)
      .toColorspace("srgb") // مطمئن می‌شیم که فرمت RGB باشه
      .raw()
      .toBuffer();
    console.log("Image buffer length after preprocessing:", imageBuffer.length); // دیباگ طول بافر

    const preprocessedData = preprocessImage(imageBuffer);

    // تبدیل به float16
    const float16Data = toFloat16Array(preprocessedData);

    // ایجاد تنسور با نوع float16
    const inputTensor = new ort.Tensor(
      "float16",
      float16Data,
      [1, 3, 640, 640]
    );

    // بارگذاری مدل
    const modelPath = path.join(process.cwd(), "models", "yolov5l.onnx");
    if (!fs.existsSync(modelPath)) {
      return NextResponse.json({ error: "مدل یافت نشد" }, { status: 404 });
    }
    const session = await ort.InferenceSession.create(modelPath);

    // اجرای مدل
    const feeds: Record<string, ort.Tensor> = {
      [session.inputNames[0]]: inputTensor,
    };
    const results = await session.run(feeds);
    console.log("Model output names:", session.outputNames); // دیباگ نام خروجی‌ها
    const output = results[session.outputNames[0]]; // فرض می‌کنیم فقط یک خروجی اصلی داره
    const data = output.data as Float32Array;

    // پردازش خروجی
    const detections = processYoloOutput(data, width!, height!);

    // ترجمه لیبل‌ها
    const result = detections.map((det) => ({
      label: det.label,
      label_fa: translateToFarsi(det.label),
      confidence: `${(det.confidence * 100).toFixed(2)}%`,
      box: det.box,
    }));

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error in image detection:", error);
    return NextResponse.json({ error: "خطای سرور" }, { status: 500 });
  }
}

function translateToFarsi(label: string): string {
  const dictionary: Record<string, string> = {
    person: "انسان",
    bicycle: "دوچرخه",
    car: "ماشین",
    motorcycle: "موتورسیکلت",
    airplane: "هواپیما",
    bus: "اتوبوس",
    train: "قطار",
    truck: "کامیون",
    boat: "قایق",
    traffic_light: "چراغ راهنمایی",
    fire_hydrant: "شیر آتش‌نشانی",
    stop_sign: "تابلو توقف",
    parking_meter: "پارکومتر",
    bench: "نیمکت",
    bird: "پرنده",
    cat: "گربه",
    dog: "سگ",
    horse: "اسب",
    sheep: "گوسفند",
    cow: "گاو",
    elephant: "فیل",
    bear: "خرس",
    zebra: "گورخر",
    giraffe: "زرافه",
    backpack: "کوله‌پشتی",
    umbrella: "چتر",
    handbag: "کیف دستی",
    tie: "کراوات",
    suitcase: "چمدان",
    frisbee: "فریزبی",
    skis: "اسکی",
    snowboard: "اسنوبرد",
    sports_ball: "توپ ورزشی",
    kite: "بادبادک",
    baseball_bat: "چوب بیسبال",
    baseball_glove: "دستکش بیسبال",
    skateboard: "اسکیت‌برد",
    surfboard: "تخته موج‌سواری",
    tennis_racket: "راکت تنیس",
    bottle: "بطری",
    wine_glass: "لیوان شراب",
    cup: "فنجان",
    fork: "چنگال",
    knife: "چاقو",
    spoon: "قاشق",
    bowl: "کاسه",
    banana: "موز",
    apple: "سیب",
    sandwich: "ساندویچ",
    orange: "پرتقال",
    broccoli: "بروکلی",
    carrot: "هویج",
    hot_dog: "هات‌داگ",
    pizza: "پیتزا",
    donut: "دونات",
    cake: "کیک",
    chair: "صندلی",
    couch: "کاناپه",
    potted_plant: "گیاه گلدانی",
    bed: "تخت",
    dining_table: "میز ناهارخوری",
    toilet: "توالت",
    tv: "تلویزیون",
    laptop: "لپ‌تاپ",
    mouse: "موس",
    remote: "کنترل",
    keyboard: "کیبورد",
    cell_phone: "تلفن همراه",
    microwave: "مایکروویو",
    oven: "فر",
    toaster: "توستر",
    sink: "سینک",
    refrigerator: "یخچال",
    book: "کتاب",
    clock: "ساعت",
    vase: "گلدان",
    scissors: "قیچی",
    teddy_bear: "عروسک خرسی",
    hair_drier: "سشوار",
    toothbrush: "مسواک",
  };
  return dictionary[label.toLowerCase()] || "نامشخص";
}
