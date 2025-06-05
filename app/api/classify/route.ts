import { NextRequest, NextResponse } from "next/server";
import * as ort from "onnxruntime-node";
import sharp from "sharp";
import fs from "fs";
import path from "path";

// مسیر فایل لیبل‌ها
const labelsPath = path.join(process.cwd(), "models", "imagenet_labels.json");
let labels: string[] = [];
try {
  console.log("Reading labels from:", labelsPath);
  labels = JSON.parse(fs.readFileSync(labelsPath, "utf-8"));
} catch (error) {
  console.error("Error reading labels:", error);
}

// تابع softmax برای تبدیل logits به احتمال
function softmax(logits: Float32Array): Float32Array {
  const maxLogit = Math.max(...logits);
  const expSum = logits.reduce((sum, val) => sum + Math.exp(val - maxLogit), 0);
  return new Float32Array(
    logits.map((val) => Math.exp(val - maxLogit) / expSum)
  );
}

// تابع نرمال‌سازی تصویر برای MobileNetV2
function preprocessImage(data: Buffer): Float32Array {
  const pixels = new Float32Array(data.length);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  // تبدیل به [0, 1] و نرمال‌سازی
  for (let i = 0; i < data.length; i += 3) {
    // RGB به ترتیب
    pixels[i] = data[i] / 255; // R
    pixels[i + 1] = data[i + 1] / 255; // G
    pixels[i + 2] = data[i + 2] / 255; // B
    // نرمال‌سازی
    pixels[i] = (pixels[i] - mean[0]) / std[0];
    pixels[i + 1] = (pixels[i + 1] - mean[1]) / std[1];
    pixels[i + 2] = (pixels[i + 2] - mean[2]) / std[2];
  }

  // تبدیل به فرمت NCHW
  const nchw = new Float32Array(1 * 3 * 224 * 224);
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const idx = y * 224 + x;
      nchw[0 * 224 * 224 + y * 224 + x] = pixels[idx * 3]; // R
      nchw[1 * 224 * 224 + y * 224 + x] = pixels[idx * 3 + 1]; // G
      nchw[2 * 224 * 224 + y * 224 + x] = pixels[idx * 3 + 2]; // B
    }
  }

  return nchw;
}

export async function POST(req: NextRequest) {
  try {
    const { filename } = await req.json();
    const imagePath = path.join(process.cwd(), "public/uploads", filename);

    // بررسی وجود فایل
    if (!fs.existsSync(imagePath)) {
      return NextResponse.json({ error: "تصویر یافت نشد" }, { status: 404 });
    }

    // خواندن و پیش‌پردازش تصویر
    const imageBuffer = await sharp(imagePath)
      .resize(224, 224)
      .removeAlpha()
      .raw()
      .toBuffer();

    const preprocessedData = preprocessImage(imageBuffer);

    // ایجاد تنسور
    const inputTensor = new ort.Tensor(
      "float32",
      preprocessedData,
      [1, 3, 224, 224]
    );

    // بارگذاری مدل
    const modelPath = path.join(process.cwd(), "models", "mobilenetv2-10.onnx");
    if (!fs.existsSync(modelPath)) {
      return NextResponse.json({ error: "مدل یافت نشد" }, { status: 404 });
    }
    const session = await ort.InferenceSession.create(modelPath);

    // اجرای مدل
    const feeds: Record<string, ort.Tensor> = {
      [session.inputNames[0]]: inputTensor,
    };
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    const data = output.data as Float32Array;

    // اعمال softmax برای تبدیل به احتمال
    const probabilities = softmax(data);

    // پیدا کردن کلاس با بالاترین احتمال
    let maxIndex = 0;
    let maxValue = probabilities[0];
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxValue) {
        maxValue = probabilities[i];
        maxIndex = i;
      }
    }

    const label = labels[maxIndex] || "unknown";
    const confidence = (maxValue * 100).toFixed(2);

    return NextResponse.json({
      label,
      label_fa: translateToFarsi(label),
      confidence: `${confidence}%`,
    });
  } catch (error) {
    console.error("Error in image classification:", error);
    return NextResponse.json({ error: "خطای سرور" }, { status: 500 });
  }
}

function translateToFarsi(label: string): string {
  const dictionary: Record<string, string> = {
    cat: "گربه",
    dog: "سگ",
    person: "انسان",
    airplane: "هواپیما",
    banana: "موز",
    apple: "سیب",
    digital_clock: "ساعت دیجیتال",
    goldfish: "ماهی طلایی",
    great_white_shark: "کوسه سفید بزرگ",
    car: "ماشین",
    tree: "درخت",
    house: "خانه",
    window_screen: "پنجره مشبک", // اضافه کردن لیبل برای window screen
    // اضافه کردن لیبل‌های بیشتر در صورت نیاز
  };

  return dictionary[label.toLowerCase()] || "نامشخص";
}
