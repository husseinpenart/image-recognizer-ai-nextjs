"use client";
import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

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
        setResult("");
      } else {
        setResult(
          `تشخیص: ${data.label} (${data.label_fa}) - احتمال: ${data.confidence}`
        );
        setError("");
      }
    } catch (err) {
      setError("خطا در ارتباط با سرور");
      setResult("");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("لطفاً یک تصویر انتخاب کنید");
      return;
    }

    setLoading(true);
    setError("");
    setResult("");

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

  return (
    <div className="p-4 max-w-md mx-auto">
      <h2 className="text-lg font-bold mb-2">تشخیص تصویر</h2>
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
      {result && <pre className="mt-4 p-2 bg-gray-100 rounded">{result}</pre>}
      {error && <p className="mt-4 text-red-500">{error}</p>}
    </div>
  );
}
