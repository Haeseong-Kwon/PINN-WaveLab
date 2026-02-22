import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import io
import os

async def main():
    print("Starting Playwright...")
    async with async_playwright() as p:
        # Taller viewport to capture the whole page
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1400, 'height': 1100})
        
        frames = []
        fps = 10

        # Create slightly smaller images to keep GIF size reasonable while capturing whole layout
        async def record_frame():
            screenshot_bytes = await page.screenshot()
            img = Image.open(io.BytesIO(screenshot_bytes))
            # Resize a bit so the GIF isn't totally gigantic
            img = img.resize((1200, 942), Image.Resampling.LANCZOS)
            frames.append(img)

        print("Navigating to dashboard...")
        await page.goto("http://localhost:3000", timeout=60000)
        
        # Initial wait for animations to finish
        for _ in range(2 * fps):
            await record_frame()
            await page.wait_for_timeout(1000 / fps)

        # Start Training
        print("Clicking Start Training...")
        start_btn = await page.wait_for_selector('button:has-text("Launch Neural Solver")', timeout=60000)
        if start_btn:
            # Hover then click
            await start_btn.hover()
            for _ in range(1 * fps): 
                await record_frame()
                await page.wait_for_timeout(1000/fps)
            await start_btn.click()
        
        # Let it run for 10 seconds (capture chart movements and waves)
        print("Running training for 10 seconds...")
        for _ in range(10 * fps):
            await record_frame()
            await page.wait_for_timeout(1000 / fps)

        # Stop training
        print("Clicking Stop...")
        stop_btn = await page.query_selector('button:has-text("Stop Simulation")')
        if stop_btn:
            await stop_btn.hover()
            for _ in range(1 * fps): 
                await record_frame()
                await page.wait_for_timeout(1000/fps)
            await stop_btn.click()

        for _ in range(1 * fps):
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
            box = await slider.bounding_box()
            if box:
                await page.mouse.click(box['x'] + box['width'] * 0.8, box['y'] + box['height'] / 2)
        
        for _ in range(1 * fps):
            await record_frame()
            await page.wait_for_timeout(1000 / fps)

        # Start again
        print("Clicking Start Training again...")
        start_btn = await page.query_selector('button:has-text("Launch Neural Solver")')
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
        # Save to assets/dashboard_full_view.gif this time, to respect the location request.
        frames[0].save('assets/dashboard_full_view.gif',
                       save_all=True, append_images=frames[1:], optimize=True, duration=int(1000/fps), loop=0)
                       
        await browser.close()
        print("GIF successfully saved at assets/dashboard_full_view.gif")

if __name__ == "__main__":
    asyncio.run(main())
