"use client";
import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<
    Array<{
      label: string;
      label_fa: string;
      confidence: string;
      box: number[];
    }>
  >([]);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const classifyImage = async (filename: string) => {
    try {
      const res = await fetch("/api/classify", {
        method: "POST",
        body: JSON.stringify({ filename }),
        headers: { "Content-Type": "application/json" },
      });

      const data = await res.json();
      if (data.error) {
        setError(data.error);
        setResult([]);
      } else {
        setResult(data);
        setError("");
      }
    } catch (err) {
      setError("خطا در ارتباط با سرور");
      setResult([]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("لطفاً یک تصویر انتخاب کنید");
      return;
    }

    setLoading(true);
    setError("");
    setResult([]);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else if (data.filename) {
        await classifyImage(data.filename);
      }
    } catch (err) {
      setError("خطا در آپلود تصویر");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (result.length > 0 && imageRef.current && canvasRef.current) {
      const img = imageRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      if (ctx) {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);

        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.font = "16px Arial";

        result.forEach((det) => {
          const [x_min, y_min, x_max, y_max] = det.box;
          ctx.strokeRect(x_min, y_min, x_max - x_min, y_max - y_min);
          ctx.fillStyle = "red";
          ctx.fillText(`${det.label_fa} (${det.confidence})`, x_min, y_min - 5);
        });
      }
    }
  }, [result]);

  return (
    <div className="p-4 max-w-md mx-auto">
      <h2 className="text-lg font-bold mb-2">تشخیص اشیا در تصویر</h2>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        className="mb-4"
      />
      <button
        onClick={handleUpload}
        disabled={loading}
        className={`p-2 w-full text-white ${
          loading ? "bg-gray-500" : "bg-blue-500 hover:bg-blue-600"
        }`}
      >
        {loading ? "در حال پردازش..." : "ارسال"}
      </button>
      {file && (
        <div className="mt-4 relative">
          <img
            ref={imageRef}
            src={URL.createObjectURL(file)}
            alt="Preview"
            className="max-w-full h-auto"
          />
          <canvas ref={canvasRef} className="absolute top-0 left-0" />
        </div>
      )}
      {result.length > 0 && (
        <pre className="mt-4 p-2 bg-gray-100 rounded">
          {result.map((det, i) => (
            <div key={i}>
              {`تشخیص: ${det.label} (${det.label_fa}) - احتمال: ${det.confidence}`}
            </div>
          ))}
        </pre>
      )}
      {error && <p className="mt-4 text-red-500">{error}</p>}
    </div>
  );
}
