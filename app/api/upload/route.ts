// app/api/upload/route.ts
import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { parseForm } from '@/app/utils/parse-form';

export const dynamic = 'force-dynamic'; // برای جلوگیری از کش‌شدن در dev

export async function POST(req: Request) {
  try {
    const { files } = await parseForm(req);

    const file = files?.file?.[0];

    if (!file) {
      return NextResponse.json({ error: 'فایلی ارسال نشده' }, { status: 400 });
    }

    const filename = path.basename(file.filepath);
    return NextResponse.json({ message: 'فایل با موفقیت آپلود شد', filename });
  } catch (err: any) {
    return NextResponse.json({ error: err.message || 'خطا در آپلود فایل' }, { status: 500 });
  }
}
