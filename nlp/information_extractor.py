"""
Information Extractor Module for Road Accident Analysis Pipeline
Uses NLP and LLM techniques to extract structured information from news articles
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from deep_translator import GoogleTranslator
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from config import (
    NLP_CONFIG, VEHICLE_TYPES, SEVERITY_LEVELS, 
    LOCATION_KEYWORDS, TRANSLATION_CONFIG
)
from utils.logger import get_pipeline_logger, log_execution_time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class InformationExtractor:
    """
    Extracts structured information from accident news articles using NLP and LLM
    """
    
    def __init__(self):
        self.logger = get_pipeline_logger()
        self.translator = GoogleTranslator(
            source=TRANSLATION_CONFIG["source_lang"],
            target=TRANSLATION_CONFIG["target_lang"]
        )
        
        # Initialize NLP models
        self.setup_nlp_models()
        
        # Load location database
        self.bangladesh_locations = self.load_bangladesh_locations()
    
    def setup_nlp_models(self):
        """Initialize NLP models for information extraction"""
        try:
            # Initialize NER pipeline for location extraction
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-base-multilingual-cased-ner-hrl",
                aggregation_strategy="simple"
            )
            
            # Initialize text classification for severity
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            self.logger.info("NLP models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not load transformer models: {str(e)}")
            self.ner_pipeline = None
            self.classifier = None
    
    def load_bangladesh_locations(self) -> Dict[str, List[str]]:
        """
        Load database of Bangladeshi locations (districts, upazilas, etc.)
        
        Returns:
            Dictionary of location categories and their names
        """
        return {
            "districts": [
                "Dhaka", "Chittagong", "Rajshahi", "Khulna", "Barisal", "Sylhet",
                "Rangpur", "Mymensingh", "Comilla", "Noakhali", "Feni", "Lakshmipur",
                "Chandpur", "Brahmanbaria", "Narayanganj", "Gazipur", "Tangail",
                "Kishoreganj", "Netrokona", "Jamalpur", "Sherpur", "Moulvibazar",
                "Habiganj", "Sunamganj", "Bogra", "Joypurhat", "Naogaon", "Natore",
                "Pabna", "Sirajganj", "Bagerhat", "Satkhira", "Jessore", "Magura",
                "Jhenaidah", "Narail", "Gopalganj", "Shariatpur", "Madaripur",
                "Rajbari", "Faridpur", "Munshiganj", "Manikganj", "Pirojpur",
                "Jhalokati", "Bhola", "Patuakhali", "Barguna", "Cox's Bazar",
                "Bandarban", "Rangamati", "Khagrachari", "Panchagarh", "Thakurgaon",
                "Dinajpur", "Nilphamari", "Lalmonirhat", "Kurigram", "Gaibandha"
            ],
            "major_roads": [
                "Dhaka-Chittagong Highway", "Dhaka-Sylhet Highway", "Dhaka-Mymensingh Highway",
                "Dhaka-Aricha Highway", "Dhaka-Tangail Highway", "Dhaka-Narayanganj Road",
                "Dhaka-Gazipur Road", "Dhaka-Savar Road", "Dhaka-Keraniganj Road",
                "Dhaka-Munshiganj Road", "Dhaka-Manikganj Road", "Dhaka-Faridpur Road"
            ]
        }
    
    def translate_text(self, text: str, source_lang: str = "bn") -> str:
        """
        Translate text from Bangla to English
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            Translated text
        """
        if not text or source_lang == "en":
            return text
        
        try:
            # Use deep-translator as primary method
            translator = GoogleTranslator(source=source_lang, target="en")
            translated = translator.translate(text)
            return translated if translated else text
        except Exception as e:
            self.logger.warning(f"Translation failed: {str(e)}")
            return text
    
    def extract_numbers(self, text: str) -> List[int]:
        """
        Extract numbers from text
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            List of extracted numbers
        """
        # Patterns for different number formats
        patterns = [
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(ব্যক্তি|person)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(আহত|injured)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(মারা গেছে|killed)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(নিহত|dead)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(মৃত|dead)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(জখম|wounded)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(আঘাত|injury)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(আহত|injured)\b',
            r'\b(\d+)\s*(জন|person|people|killed|dead|died|injured|wounded)\s*(জখম|wounded)\b',
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    numbers.append(int(match[0]))
                except (ValueError, IndexError):
                    continue
        
        return numbers
    
    def extract_location(self, text: str, language: str) -> Dict[str, str]:
        """
        Extract location information from text
        
        Args:
            text: Text to extract location from
            language: Language of the text
            
        Returns:
            Dictionary with location information
        """
        location_info = {
            'district': '',
            'upazila': '',
            'road': '',
            'specific_location': ''
        }
        
        # Use NER if available
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text[:512])  # Limit text length
                for entity in entities:
                    if entity['score'] > NLP_CONFIG["confidence_threshold"]:
                        if entity['entity_group'] == 'LOC':
                            location_info['specific_location'] = entity['word']
                        elif entity['entity_group'] == 'GPE':
                            location_info['district'] = entity['word']
            except Exception as e:
                self.logger.debug(f"NER extraction failed: {str(e)}")
        
        # Rule-based extraction
        text_lower = text.lower()
        
        # Check for district names
        for district in self.bangladesh_locations["districts"]:
            if district.lower() in text_lower:
                location_info['district'] = district
                break
        
        # Check for major roads
        for road in self.bangladesh_locations["major_roads"]:
            if road.lower() in text_lower:
                location_info['road'] = road
                break
        
        # Check for location keywords
        location_keywords = LOCATION_KEYWORDS.get(language, [])
        for keyword in location_keywords:
            if keyword.lower() in text_lower:
                # Extract text around the keyword
                keyword_index = text_lower.find(keyword.lower())
                if keyword_index != -1:
                    start = max(0, keyword_index - 50)
                    end = min(len(text), keyword_index + len(keyword) + 50)
                    location_info['specific_location'] = text[start:end].strip()
                break
        
        return location_info
    
    def extract_vehicle_types(self, text: str, language: str) -> List[str]:
        """
        Extract vehicle types involved in the accident
        
        Args:
            text: Text to extract vehicle types from
            language: Language of the text
            
        Returns:
            List of vehicle types found
        """
        vehicle_types = []
        text_lower = text.lower()
        
        vehicle_dict = VEHICLE_TYPES.get(language, {})
        
        for vehicle_category, keywords in vehicle_dict.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    vehicle_types.append(vehicle_category)
                    break
        
        return list(set(vehicle_types))  # Remove duplicates
    
    def classify_severity(self, text: str, fatalities: int, injuries: int) -> str:
        """
        Classify accident severity based on text and casualty numbers
        
        Args:
            text: Text describing the accident
            fatalities: Number of fatalities
            injuries: Number of injuries
            
        Returns:
            Severity classification
        """
        text_lower = text.lower()
        
        # Check for severity keywords
        for severity, keywords in SEVERITY_LEVELS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return severity
        
        # Classify based on casualty numbers
        if fatalities > 0:
            return "fatal"
        elif injuries > 5:
            return "major"
        else:
            return "minor"
    
    def extract_date(self, text: str, parsed_date: datetime = None) -> Optional[datetime]:
        """
        Extract or validate date from text
        
        Args:
            text: Text containing date information
            parsed_date: Already parsed date object
            
        Returns:
            Parsed datetime object
        """
        if parsed_date:
            return parsed_date
        
        # Additional date patterns
        patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
            r'(\d{1,2})\s+(জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)\s+(\d{4})',
            r'(আজ|today|yesterday|কাল|tomorrow)',
            r'(\d{1,2})\s+(hours?|days?|weeks?|months?)\s+ago'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if 'জানুয়ারি' in pattern:
                        day, month_bn, year = match.groups()
                        month_map = {
                            'জানুয়ারি': 1, 'ফেব্রুয়ারি': 2, 'মার্চ': 3, 'এপ্রিল': 4,
                            'মে': 5, 'জুন': 6, 'জুলাই': 7, 'আগস্ট': 8,
                            'সেপ্টেম্বর': 9, 'অক্টোবর': 10, 'নভেম্বর': 11, 'ডিসেম্বর': 12
                        }
                        month = month_map.get(month_bn, 1)
                        return datetime(int(year), month, int(day))
                    elif 'ago' in pattern:
                        # Handle relative dates
                        number, unit = match.groups()
                        number = int(number)
                        if 'hour' in unit:
                            return datetime.now() - timedelta(hours=number)
                        elif 'day' in unit:
                            return datetime.now() - timedelta(days=number)
                        elif 'week' in unit:
                            return datetime.now() - timedelta(weeks=number)
                        elif 'month' in unit:
                            return datetime.now() - timedelta(days=number*30)
                    else:
                        groups = match.groups()
                        if len(groups[0]) == 4:  # YYYY/MM/DD
                            return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                        else:  # DD/MM/YYYY
                            return datetime(int(groups[2]), int(groups[1]), int(groups[0]))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    @log_execution_time
    def extract_information(self, article: Dict) -> Dict[str, Any]:
        """
        Extract structured information from a news article
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            Dictionary with extracted structured information
        """
        title = article.get('title', '')
        text = article.get('text', '')
        language = article.get('language', 'english')
        source_url = article.get('url', '')
        source_name = article.get('source', '')
        
        # Combine title and text for analysis
        full_text = f"{title} {text}"
        
        # Translate if needed
        if language == 'bangla':
            translated_text = self.translate_text(full_text, 'bn')
        else:
            translated_text = full_text
        
        # Extract numbers (fatalities and injuries)
        numbers = self.extract_numbers(full_text)
        fatalities = numbers[0] if len(numbers) > 0 else 0
        injuries = numbers[1] if len(numbers) > 1 else 0
        
        # Extract location information
        location_info = self.extract_location(full_text, language)
        
        # Extract vehicle types
        vehicle_types = self.extract_vehicle_types(full_text, language)
        
        # Classify severity
        severity = self.classify_severity(full_text, fatalities, injuries)
        
        # Extract/validate date
        date = self.extract_date(full_text, article.get('parsed_date'))
        
        # Create structured output
        extracted_info = {
            'date': date,
            'title': title,
            'text': text[:500],  # Truncate for storage
            'translated_text': translated_text[:500],
            'district': location_info['district'],
            'upazila': location_info['upazila'],
            'road': location_info['road'],
            'specific_location': location_info['specific_location'],
            'fatalities': fatalities,
            'injuries': injuries,
            'vehicle_types': ','.join(vehicle_types) if vehicle_types else '',
            'severity': severity,
            'source_url': source_url,
            'source_name': source_name,
            'language': language,
            'extraction_confidence': 0.8  # Default confidence
        }
        
        return extracted_info
    
    @log_execution_time
    def extract_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Extract information from a batch of articles
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of extracted information dictionaries
        """
        extracted_data = []
        
        for i, article in enumerate(articles):
            try:
                extracted_info = self.extract_information(article)
                extracted_data.append(extracted_info)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(articles)} articles")
                    
            except Exception as e:
                self.logger.error(f"Error extracting information from article {i}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully extracted information from {len(extracted_data)} articles")
        return extracted_data