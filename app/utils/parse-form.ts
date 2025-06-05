// utils/parse-form.ts
import { IncomingForm } from 'formidable';
import { Readable } from 'stream';

export const config = {
  api: {
    bodyParser: false,
  },
};

export function parseForm(req: Request): Promise<{ files: any }> {
  const form = new IncomingForm({
    uploadDir: process.cwd() + '/public/uploads',
    keepExtensions: true,
    multiples: true,
  });

  return new Promise((resolve, reject) => {
    const nodeReq = Readable.fromWeb(req.body as any) as any;
    nodeReq.headers = Object.fromEntries(req.headers.entries());

    form.parse(nodeReq, (err, fields, files) => {
      if (err) return reject(err);
      resolve({ files });
    });
  });
}
