import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { parseForm } from "@/app/utils/parse-form";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  try {
    console.log("Upload request received");

    // Create uploads directory if it doesn't exist
    const uploadsDir = path.join(process.cwd(), "public/uploads");
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
      console.log("Created uploads directory:", uploadsDir);
    }

    const { files } = await parseForm(req);
    console.log("Parse form result:", files);

    const file = files?.file?.[0];

    if (!file) {
      console.error("No file found in request");
      return NextResponse.json({ error: "فایلی ارسال نشده" }, { status: 400 });
    }

    // Get just the filename from the filepath
    const filename = path.basename(file.filepath);

    console.log("File uploaded successfully:", filename);
    console.log("File path:", file.filepath);
    console.log("File size:", file.size, "bytes");
    console.log("File type:", file.mimetype);

    return NextResponse.json({
      message: "فایل با موفقیت آپلود شد",
      filename,
    });
  } catch (err: any) {
    console.error("Upload error:", err);
    return NextResponse.json(
      {
        error: err.message || "خطا در آپلود فایل",
      },
      { status: 500 }
    );
  }
}
