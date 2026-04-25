import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

import asyncio
import httpx
from services.generation import _generate_garment_only

async def main():
    async with httpx.AsyncClient() as client:
        print("Generating garment...")
        try:
            res = await _generate_garment_only(client, "t-shirt", "red", "casual", "male")
            print(f"Generated {len(res)} bytes")
        except Exception as e:
            print(f"Error: {e}")

asyncio.run(main())
