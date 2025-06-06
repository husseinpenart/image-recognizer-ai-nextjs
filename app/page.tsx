"use client";
import Link from "next/link";
import { useState, useRef, useEffect } from "react";

interface Detection {
  label: string;
  label_fa: string;
  confidence: string;
  box: number[];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Detection[]>([]);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  const classifyImage = async (filename: string) => {
    try {
      console.log("Classifying image:", filename);
      const res = await fetch("/api/classify", {
        method: "POST",
        body: JSON.stringify({ filename }),
        headers: { "Content-Type": "application/json" },
      });

      const data = await res.json();
      console.log("Classification response:", data);

      if (data.error) {
        setError(data.error);
        setResult([]);
      } else {
        setResult(Array.isArray(data) ? data : []);
        setError("");
      }
    } catch (err) {
      console.error("Classification error:", err);
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

  // Draw bounding boxes when image is loaded or results change
  useEffect(() => {
    if (
      result.length > 0 &&
      imageRef.current &&
      canvasRef.current &&
      imageLoaded
    ) {
      const img = imageRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      if (ctx) {
        // Set canvas size to match image display size
        const rect = img.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Calculate scale factors
        const scaleX = rect.width / img.naturalWidth;
        const scaleY = rect.height / img.naturalHeight;

        // Sort detections by confidence (highest first) and limit to top 10
        const topDetections = [...result]
          .sort((a, b) => {
            const confA = Number.parseFloat(a.confidence.replace("%", ""));
            const confB = Number.parseFloat(b.confidence.replace("%", ""));
            return confB - confA;
          })
          .slice(0, 10);

        // Draw bounding boxes for top detections
        topDetections.forEach((det) => {
          const [x_min, y_min, x_max, y_max] = det.box;

          // Scale coordinates to canvas size
          const scaledX = x_min * scaleX;
          const scaledY = y_min * scaleY;
          const scaledWidth = (x_max - x_min) * scaleX;
          const scaledHeight = (y_max - y_min) * scaleY;

          // Draw bounding box
          ctx.strokeStyle = "#FF0000";
          ctx.lineWidth = 2;
          ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

          // Draw label with background
          const label = `${det.label_fa} (${det.confidence})`;
          const textMetrics = ctx.measureText(label);
          const textHeight = 16;

          // Draw background for text
          ctx.fillStyle = "rgba(255, 0, 0, 0.8)";
          ctx.fillRect(
            scaledX,
            scaledY - textHeight - 2,
            textMetrics.width + 4,
            textHeight + 4
          );

          // Draw text
          ctx.fillStyle = "#FFFFFF";
          ctx.font = "14px Arial";
          ctx.fillText(label, scaledX + 2, scaledY - 4);
        });
      }
    }
  }, [result, imageLoaded]);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <Link href={"/Model"} className="decoration-1 text-blue-500">my Trained model Link</Link>
      <h1 className="text-2xl font-bold mb-6 text-center">
        تشخیص اشیا در تصویر
      </h1>
    
      <div className="mb-6">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const selectedFile = e.target.files?.[0] ?? null;
            setFile(selectedFile);
            setImageLoaded(false); // Reset image loaded state when new file is selected
            setResult([]); // Clear previous results
          }}
          className="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />

        <button
          onClick={handleUpload}
          disabled={loading || !file}
          className={`w-full py-3 px-4 rounded-lg text-white font-medium ${
            loading || !file
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700 active:bg-blue-800"
          } transition-colors`}
        >
          {loading ? "در حال پردازش..." : "تشخیص اشیا"}
        </button>
      </div>

      {file && (
        <div className="mb-6 relative">
          <img
            ref={imageRef}
            src={URL.createObjectURL(file) || "/placeholder.svg"}
            alt="Preview"
            className="max-w-full h-auto rounded-lg shadow-lg"
            onLoad={() => setImageLoaded(true)}
            style={{ maxHeight: "500px", objectFit: "contain" }}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 pointer-events-none"
            style={{ maxHeight: "500px", width: "100%", height: "100%" }}
          />
        </div>
      )}

      {result.length > 0 && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">نتایج تشخیص:</h3>
          <div className="space-y-2">
            {/* Show only top 10 detections sorted by confidence */}
            {[...result]
              .sort((a, b) => {
                const confA = Number.parseFloat(a.confidence.replace("%", ""));
                const confB = Number.parseFloat(b.confidence.replace("%", ""));
                return confB - confA;
              })
              .slice(0, 10)
              .map((det, i) => (
                <div key={i} className="bg-white p-3 rounded border">
                  <div className="font-medium">
                    {det.label} ({det.label_fa})
                  </div>
                  <div className="text-sm text-gray-600">
                    احتمال: {det.confidence}
                  </div>
                  <div className="text-xs text-gray-500">
                    مختصات: [
                    {det.box.map((coord) => Math.round(coord)).join(", ")}]
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700">{error}</p>
        </div>
      )}
    </div>
  );
}
