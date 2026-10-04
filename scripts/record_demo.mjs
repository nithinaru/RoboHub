import { createRequire } from "node:module";
const require = createRequire("/tmp/node_modules/playwright/package.json");
const { chromium } = require("playwright");
import { mkdir } from "node:fs/promises";

const out = new URL("../brag-output-2026-10-03-154200/screen/", import.meta.url);
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: "chrome" });
const context = await browser.newContext({
  viewport: { width: 1600, height: 1000 },
  deviceScaleFactor: 1,
  recordVideo: { dir: out.pathname, size: { width: 1600, height: 1000 } },
});
const page = await context.newPage();
await page.goto("http://127.0.0.1:8777/?demo=1", { waitUntil: "domcontentloaded" });
await page.waitForTimeout(76000);
await context.close();
await browser.close();
console.log("recorded");
