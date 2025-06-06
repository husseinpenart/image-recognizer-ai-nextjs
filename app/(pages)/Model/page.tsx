"use client";
import React, { useEffect, useRef, useState } from "react";

const page = () => {
  const [result, setResult] = useState<string>(""); // برای ذخیره نتیجه تشخیص
  const canvasRef = useRef<HTMLCanvasElement>(null); // برای دسترسی به canvas
  const [imageSrc, setImageSrc] = useState<string>(""); // برای انتخاب تصویر

  // تابع برای پردازش تصویر
  const processImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // بارگذاری تصویر
    const img = new Image();
    img.src = imageSrc; // مسیر تصویر انتخاب‌شده
    img.onload = () => {
      // تنظیم اندازه canvas به اندازه تصویر
      canvas.width = img.width;
      canvas.height = img.height;

      // کشیدن تصویر روی canvas
      ctx.drawImage(img, 0, 0);

      // گرفتن داده‌های پیکسل
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      const data = imageData.data; // آرایه‌ای از مقادیر RGBA

      // شمارش پیکسل‌های روشن
      let brightPixelCount = 0;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i]; // قرمز
        const g = data[i + 1]; // سبز
        const b = data[i + 2]; // آبی
        // اگه مقدار RGB بالا باشه (روشن)، می‌شمریمش
        if (r > 200 && g > 200 && b > 200) {
          brightPixelCount++;
        }
      }
      console.log("Processing image:", imageSrc);
      // تصمیم‌گیری: اگه تعداد پیکسل‌های روشن کم باشه، دایره فرض می‌کنیم
      const totalPixels = img.width * img.height;
      const brightRatio = brightPixelCount / totalPixels;
      console.log("Bright pixel ratio:", brightRatio); // برای دیباگ

      if (brightRatio < 0.75) {
        setResult("این یه دایره است!");
      } else {
        setResult("این یه مربع است!");
      }
    };
  };

  // وقتی تصویر انتخاب می‌شه، تابع پردازش رو اجرا کن
  useEffect(() => {
    if (imageSrc) {
      processImage();
    }
  }, [imageSrc]);

  return (
    <div style={{ padding: "20px", textAlign: "center" }}>
      <h1>تشخیص شکل‌های هندسی</h1>
      <p>یه تصویر از دایره یا مربع انتخاب کن:</p>
      <select onChange={(e) => setImageSrc(e.target.value)} value={imageSrc}>
        <option value="">یه تصویر انتخاب کن</option>
        <option value="/shapes/square.png">دایره ۱</option>
        <option value="/shapes/circle.png">مربع ۱</option>
      </select>
      <br />
      <br />
      {imageSrc && (
        <img src={imageSrc} alt="شکل" style={{ maxWidth: "200px" }} />
      )}
      <br />
      <br />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      {result && <p>{result}</p>}
    </div>
  );
};

export default page;
