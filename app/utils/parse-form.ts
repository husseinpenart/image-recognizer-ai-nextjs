import { IncomingForm } from "formidable";
import { Readable } from "stream";
import fs from "fs";
import path from "path";

export const config = {
  api: {
    bodyParser: false,
  },
};

export function parseForm(req: Request): Promise<{ files: any }> {
  // Create uploads directory if it doesn't exist
  const uploadsDir = path.join(process.cwd(), "public/uploads");
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }

  const form = new IncomingForm({
    uploadDir: uploadsDir,
    keepExtensions: true,
    multiples: true,
  });

  console.log("Upload directory:", uploadsDir);

  return new Promise((resolve, reject) => {
    const nodeReq = Readable.fromWeb(req.body as any) as any;
    nodeReq.headers = Object.fromEntries(req.headers.entries());

    form.parse(nodeReq, (err, fields, files) => {
      if (err) {
        console.error("Form parse error:", err);
        return reject(err);
      }
      console.log("Uploaded files:", files);
      resolve({ files });
    });
  });
}
