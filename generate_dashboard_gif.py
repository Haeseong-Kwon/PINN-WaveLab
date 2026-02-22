import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import io
import os

async def main():
    print("Starting Playwright...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1280, 'height': 800})
        
        print("Navigating to dashboard...")
        await page.goto("http://localhost:3000", timeout=60000)
        
        # Wait for the app to load
        print("Waiting for Start Training button to appear...")
        start_btn = await page.wait_for_selector('button:has-text("Start Training")', timeout=60000)
        
        print("Clicking Start Training...")
        if start_btn:
            await start_btn.click()
            print("Clicked Start Training button.")
        else:
            print("Start Training button not found, recording anyway...")
            
        # Give it a second to connect via WS and start the training
        await page.wait_for_timeout(2000)
        
        frames = []
        fps = 10
        duration = 15 # 15 seconds
        
        print(f"Recording {duration} seconds at {fps} FPS...")
        for i in range(duration * fps):
            if i % 10 == 0:
                print(f"Recorded {i}/{duration * fps} frames...")
            screenshot_bytes = await page.screenshot()
            frames.append(Image.open(io.BytesIO(screenshot_bytes)))
            await page.wait_for_timeout(1000 / fps)
            
        print("Converting and saving GIF...")
        os.makedirs("assets", exist_ok=True)
        # Using Pillow to save GIF
        frames[0].save('assets/dashboard_full_view.gif',
                       save_all=True, append_images=frames[1:], optimize=True, duration=int(1000/fps), loop=0)
                       
        await browser.close()
        print("GIF successfully saved at assets/dashboard_full_view.gif")

if __name__ == "__main__":
    asyncio.run(main())
