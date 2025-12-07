#!/usr/bin/env node
/**
 * Generate screenshot of README using Playwright
 * Usage: node screenshot_readme.js <url> <output_path>
 */

const { chromium } = require('playwright');

const URL = process.argv[2];
const OUTPUT_PATH = process.argv[3];

if (!URL || !OUTPUT_PATH) {
  console.error('Usage: node screenshot_readme.js <url> <output_path>');
  process.exit(1);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000); // Wait for any dynamic content
  await page.screenshot({ path: OUTPUT_PATH, fullPage: true });
  await browser.close();
  console.log(`Screenshot saved to: ${OUTPUT_PATH}`);
})().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});

