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
        
        frames = []
        fps = 10

        async def record_frame():
            screenshot_bytes = await page.screenshot()
            frames.append(Image.open(io.BytesIO(screenshot_bytes)))

        print("Navigating to dashboard...")
        await page.goto("http://localhost:3000", timeout=60000)
        
        # Initial wait
        for _ in range(2 * fps):
            await record_frame()
            await page.wait_for_timeout(1000 / fps)

        # Start Training
        print("Clicking Start Training...")
        start_btn = await page.wait_for_selector('button:has-text("Start Training")', timeout=60000)
        if start_btn:
            # Hover then click
            await start_btn.hover()
            for _ in range(1 * fps): 
                await record_frame()
                await page.wait_for_timeout(1000/fps)
            await start_btn.click()
        
        # Let it run for 10 seconds
        print("Running training for 10 seconds...")
        for _ in range(10 * fps):
            await record_frame()
            await page.wait_for_timeout(1000 / fps)

        # Stop training
        print("Clicking Stop...")
        stop_btn = await page.query_selector('button:has-text("Stop")')
        if stop_btn:
            await stop_btn.hover()
            for _ in range(1 * fps): 
                await record_frame()
                await page.wait_for_timeout(1000/fps)
            await stop_btn.click()

        for _ in range(2 * fps):
            await record_frame()
            await page.wait_for_timeout(1000 / fps)

        # Change Parameter
        print("Changing Wave Number...")
        slider = await page.query_selector('input[type="range"]')
        if slider:
            await slider.hover()
            for _ in range(1 * fps): 
                await record_frame()
                await page.wait_for_timeout(1000/fps)
            # Click near the right side of the slider
            box = await slider.bounding_box()
            if box:
                await page.mouse.click(box['x'] + box['width'] * 0.8, box['y'] + box['height'] / 2)
        
        for _ in range(1 * fps):
            await record_frame()
            await page.wait_for_timeout(1000 / fps)

        # Start again
        print("Clicking Start Training again...")
        start_btn = await page.query_selector('button:has-text("Start Training")')
        if start_btn:
            await start_btn.hover()
            for _ in range(1 * fps): 
                await record_frame()
                await page.wait_for_timeout(1000/fps)
            await start_btn.click()

        # Let it run for 8 seconds
        print("Running new training for 8 seconds...")
        for _ in range(8 * fps):
            await record_frame()
            await page.wait_for_timeout(1000 / fps)
            
        print("Converting and saving GIF...")
        # Save to the root layout as requested by the user
        frames[0].save('pinn_wavelab_demo.gif',
                       save_all=True, append_images=frames[1:], optimize=True, duration=int(1000/fps), loop=0)
                       
        await browser.close()
        print("GIF successfully saved at pinn_wavelab_demo.gif")

if __name__ == "__main__":
    asyncio.run(main())
