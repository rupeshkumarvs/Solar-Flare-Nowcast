import os
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
