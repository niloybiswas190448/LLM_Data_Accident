"""
Configuration file for Road Accident Analysis Pipeline
Contains all settings, URLs, and parameters for the automated system
"""

import os
from datetime import datetime, timedelta

# Database Configuration
DATABASE_PATH = "road_accidents.db"
CSV_OUTPUT_PATH = "accident_data.csv"

# News Sources Configuration
NEWS_SOURCES = {
    "prothom_alo": {
        "base_url": "https://www.prothomalo.com",
        "search_url": "https://www.prothomalo.com/search?q={query}",
        "language": "bangla",
        "selectors": {
            "article_links": "h3 a, .headline a",
            "title": "h1, .headline",
            "content": ".content, .article-content",
            "date": ".time, .published-date",
            "pagination": ".pagination a"
        }
    },
    "daily_star": {
        "base_url": "https://www.thedailystar.net",
        "search_url": "https://www.thedailystar.net/search?q={query}",
        "language": "english",
        "selectors": {
            "article_links": ".article-title a, h3 a",
            "title": "h1, .article-title",
            "content": ".article-content, .content",
            "date": ".published-date, .date",
            "pagination": ".pagination a"
        }
    },
    "ittefaq": {
        "base_url": "https://www.ittefaq.com.bd",
        "search_url": "https://www.ittefaq.com.bd/search?q={query}",
        "language": "bangla",
        "selectors": {
            "article_links": ".headline a, h3 a",
            "title": "h1, .headline",
            "content": ".content, .article-content",
            "date": ".time, .published-date",
            "pagination": ".pagination a"
        }
    }
}

# Keywords for accident detection (in both Bangla and English)
ACCIDENT_KEYWORDS = {
    "bangla": [
        "দুর্ঘটনা", "আঘাত", "মৃত্যু", "জখম", "রাস্তা", "গাড়ি", "বাস", "ট্রাক",
        "মোটরসাইকেল", "মারা গেছে", "আহত", "ধাক্কা", "সংঘর্ষ", "পড়ে গেছে",
        "ট্রাফিক", "সড়ক", "হাইওয়ে", "মহাসড়ক", "ক্রাশ", "অ্যাক্সিডেন্ট"
    ],
    "english": [
        "accident", "crash", "collision", "killed", "injured", "road", "car",
        "bus", "truck", "motorcycle", "traffic", "highway", "fatal", "death",
        "wounded", "hit", "run over", "overturned", "burned", "exploded"
    ]
}

# Location keywords for extraction
LOCATION_KEYWORDS = {
    "bangla": [
        "জেলায়", "উপজেলায়", "থানায়", "মহাসড়কে", "রাস্তায়", "সড়কে",
        "হাইওয়েতে", "ব্রিজে", "ফ্লাইওভারে", "ইন্টারসেকশনে", "চৌরাস্তায়"
    ],
    "english": [
        "district", "upazila", "thana", "highway", "road", "street",
        "bridge", "flyover", "intersection", "crossing", "area"
    ]
}

# Vehicle types for classification
VEHICLE_TYPES = {
    "bangla": {
        "bus": ["বাস", "মিনিবাস", "লোকাল বাস"],
        "truck": ["ট্রাক", "লরি", "পিকআপ"],
        "car": ["গাড়ি", "কার", "মাইক্রো", "সিএনজি"],
        "motorcycle": ["মোটরসাইকেল", "বাইক", "সাইকেল"],
        "rickshaw": ["রিকশা", "অটোরিকশা"],
        "train": ["ট্রেন", "রেলগাড়ি"]
    },
    "english": {
        "bus": ["bus", "minibus", "local bus"],
        "truck": ["truck", "lorry", "pickup"],
        "car": ["car", "micro", "cng"],
        "motorcycle": ["motorcycle", "bike", "cycle"],
        "rickshaw": ["rickshaw", "auto-rickshaw"],
        "train": ["train", "railway"]
    }
}

# Severity classification
SEVERITY_LEVELS = {
    "minor": ["minor", "light", "ছোট", "হালকা"],
    "major": ["major", "serious", "বড়", "গুরুতর"],
    "fatal": ["fatal", "death", "killed", "মৃত্যু", "মারা গেছে"]
}

# NLP/LLM Configuration
NLP_CONFIG = {
    "model_name": "bert-base-multilingual-cased",  # For multilingual support
    "max_length": 512,
    "batch_size": 8,
    "confidence_threshold": 0.7
}

# Scraping Configuration
SCRAPING_CONFIG = {
    "max_articles_per_source": 100,
    "delay_between_requests": 2,  # seconds
    "timeout": 30,
    "retry_attempts": 3,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Date range for analysis (last 5 years)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5*365)

# Analysis Configuration
ANALYSIS_CONFIG = {
    "trend_period": "monthly",  # monthly, yearly, weekly
    "top_locations_count": 10,
    "visualization_format": "png",
    "output_directory": "outputs"
}

# Translation Configuration
TRANSLATION_CONFIG = {
    "source_lang": "bn",
    "target_lang": "en",
    "fallback_service": "google"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "road_accident_analysis.log"
}

# Create output directory if it doesn't exist
os.makedirs(ANALYSIS_CONFIG["output_directory"], exist_ok=True)