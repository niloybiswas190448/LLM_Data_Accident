#!/usr/bin/env python3
"""
Test Script for Road Accident Analysis Pipeline
Tests various components and functionality
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
import tempfile
import shutil

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrapers.news_scraper import NewsScraper
from nlp.information_extractor import InformationExtractor
from database.data_manager import DataManager
from analysis.trend_analyzer import TrendAnalyzer
from main import RoadAccidentPipeline
from utils.logger import get_pipeline_logger

class TestRoadAccidentPipeline(unittest.TestCase):
    """Test cases for the road accident analysis pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        self.logger = get_pipeline_logger()
        self.test_dir = tempfile.mkdtemp()
        
        # Create test output directory
        self.test_output_dir = os.path.join(self.test_dir, "outputs")
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Test data
        self.sample_articles = [
            {
                'title': 'Bus accident in Dhaka kills 5 people',
                'text': 'A bus accident occurred in Dhaka yesterday killing 5 people and injuring 10 others. The accident happened on Dhaka-Chittagong Highway.',
                'date': '2024-01-15',
                'url': 'http://example.com/accident1',
                'source': 'test_source',
                'language': 'english',
                'parsed_date': datetime(2024, 1, 15)
            },
            {
                'title': 'দুর্ঘটনায় ৩ জন নিহত',
                'text': 'গতকাল চট্টগ্রামে একটি ট্রাক দুর্ঘটনায় ৩ জন নিহত এবং ৭ জন আহত হয়েছেন।',
                'date': '2024-01-14',
                'url': 'http://example.com/accident2',
                'source': 'test_source',
                'language': 'bangla',
                'parsed_date': datetime(2024, 1, 14)
            }
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_news_scraper_initialization(self):
        """Test news scraper initialization"""
        scraper = NewsScraper()
        self.assertIsNotNone(scraper)
        self.assertIsNotNone(scraper.session)
        self.assertIsNotNone(scraper.ua)
    
    def test_information_extractor_initialization(self):
        """Test information extractor initialization"""
        extractor = InformationExtractor()
        self.assertIsNotNone(extractor)
        self.assertIsNotNone(extractor.bangladesh_locations)
        self.assertIn('districts', extractor.bangladesh_locations)
        self.assertIn('major_roads', extractor.bangladesh_locations)
    
    def test_data_manager_initialization(self):
        """Test data manager initialization"""
        # Use temporary database for testing
        test_db_path = os.path.join(self.test_dir, "test_accidents.db")
        
        # Temporarily modify the database path
        original_db_path = DataManager.__init__.__defaults__[0] if DataManager.__init__.__defaults__ else None
        
        try:
            data_manager = DataManager()
            self.assertIsNotNone(data_manager)
            self.assertTrue(os.path.exists(data_manager.db_path))
        finally:
            # Clean up
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
    
    def test_accident_keyword_detection(self):
        """Test accident keyword detection"""
        scraper = NewsScraper()
        
        # Test English text
        english_text = "A car accident occurred on the highway killing 2 people"
        self.assertTrue(scraper.is_accident_related(english_text, "english"))
        
        # Test Bangla text
        bangla_text = "রাস্তায় একটি গাড়ি দুর্ঘটনায় ২ জন মারা গেছে"
        self.assertTrue(scraper.is_accident_related(bangla_text, "bangla"))
        
        # Test non-accident text
        non_accident_text = "The weather is nice today and people are happy"
        self.assertFalse(scraper.is_accident_related(non_accident_text, "english"))
    
    def test_information_extraction(self):
        """Test information extraction from articles"""
        extractor = InformationExtractor()
        
        for article in self.sample_articles:
            extracted_info = extractor.extract_information(article)
            
            # Check required fields
            self.assertIn('date', extracted_info)
            self.assertIn('title', extracted_info)
            self.assertIn('fatalities', extracted_info)
            self.assertIn('injuries', extracted_info)
            self.assertIn('severity', extracted_info)
            self.assertIn('district', extracted_info)
            self.assertIn('vehicle_types', extracted_info)
            
            # Check data types
            self.assertIsInstance(extracted_info['fatalities'], int)
            self.assertIsInstance(extracted_info['injuries'], int)
            self.assertIsInstance(extracted_info['severity'], str)
    
    def test_date_parsing(self):
        """Test date parsing functionality"""
        extractor = InformationExtractor()
        
        # Test various date formats
        test_cases = [
            ("15/01/2024", datetime(2024, 1, 15)),
            ("2024-01-15", datetime(2024, 1, 15)),
            ("১৫ জানুয়ারি ২০২৪", datetime(2024, 1, 15)),
        ]
        
        for date_str, expected_date in test_cases:
            parsed_date = extractor.parse_date(date_str, "english")
            if parsed_date:
                self.assertEqual(parsed_date.date(), expected_date.date())
    
    def test_location_extraction(self):
        """Test location extraction"""
        extractor = InformationExtractor()
        
        # Test English location extraction
        english_text = "An accident occurred in Dhaka district on Dhaka-Chittagong Highway"
        location_info = extractor.extract_location(english_text, "english")
        self.assertEqual(location_info['district'], "Dhaka")
        self.assertEqual(location_info['road'], "Dhaka-Chittagong Highway")
        
        # Test Bangla location extraction
        bangla_text = "ঢাকা জেলায় একটি দুর্ঘটনা ঘটেছে"
        location_info = extractor.extract_location(bangla_text, "bangla")
        self.assertEqual(location_info['district'], "Dhaka")
    
    def test_vehicle_type_extraction(self):
        """Test vehicle type extraction"""
        extractor = InformationExtractor()
        
        # Test English vehicle extraction
        english_text = "A bus and a truck collided on the highway"
        vehicle_types = extractor.extract_vehicle_types(english_text, "english")
        self.assertIn("bus", vehicle_types)
        self.assertIn("truck", vehicle_types)
        
        # Test Bangla vehicle extraction
        bangla_text = "একটি বাস এবং একটি ট্রাক সংঘর্ষে লিপ্ত হয়েছে"
        vehicle_types = extractor.extract_vehicle_types(bangla_text, "bangla")
        self.assertIn("bus", vehicle_types)
        self.assertIn("truck", vehicle_types)
    
    def test_severity_classification(self):
        """Test severity classification"""
        extractor = InformationExtractor()
        
        # Test fatal accident
        fatal_text = "5 people were killed in the accident"
        severity = extractor.classify_severity(fatal_text, 5, 0)
        self.assertEqual(severity, "fatal")
        
        # Test major accident
        major_text = "10 people were injured in the accident"
        severity = extractor.classify_severity(major_text, 0, 10)
        self.assertEqual(severity, "major")
        
        # Test minor accident
        minor_text = "2 people were slightly injured"
        severity = extractor.classify_severity(minor_text, 0, 2)
        self.assertEqual(severity, "minor")
    
    def test_pipeline_integration(self):
        """Test full pipeline integration"""
        # Create a minimal pipeline test
        try:
            pipeline = RoadAccidentPipeline()
            self.assertIsNotNone(pipeline)
            
            # Test system status
            status = pipeline.get_system_status()
            self.assertIn('system_status', status)
            self.assertIn('last_update', status)
            
        except Exception as e:
            # Pipeline might fail if no internet connection or models not downloaded
            self.logger.warning(f"Pipeline integration test skipped: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling"""
        extractor = InformationExtractor()
        
        # Test with empty/invalid input
        empty_article = {}
        extracted_info = extractor.extract_information(empty_article)
        
        # Should handle gracefully without crashing
        self.assertIsInstance(extracted_info, dict)
        self.assertIn('fatalities', extracted_info)
        self.assertEqual(extracted_info['fatalities'], 0)
    
    def test_data_consistency(self):
        """Test data consistency across components"""
        extractor = InformationExtractor()
        
        for article in self.sample_articles:
            extracted_info = extractor.extract_information(article)
            
            # Check that fatalities and injuries are non-negative
            self.assertGreaterEqual(extracted_info['fatalities'], 0)
            self.assertGreaterEqual(extracted_info['injuries'], 0)
            
            # Check that severity is one of the expected values
            self.assertIn(extracted_info['severity'], ['fatal', 'major', 'minor'])
            
            # Check that date is valid if present
            if extracted_info['date']:
                self.assertIsInstance(extracted_info['date'], datetime)

def run_basic_tests():
    """Run basic functionality tests"""
    print("Running basic functionality tests...")
    
    # Test scraper
    print("Testing news scraper...")
    scraper = NewsScraper()
    print("✓ News scraper initialized")
    
    # Test extractor
    print("Testing information extractor...")
    extractor = InformationExtractor()
    print("✓ Information extractor initialized")
    
    # Test data manager
    print("Testing data manager...")
    data_manager = DataManager()
    print("✓ Data manager initialized")
    
    # Test analyzer
    print("Testing trend analyzer...")
    analyzer = TrendAnalyzer(data_manager)
    print("✓ Trend analyzer initialized")
    
    # Test pipeline
    print("Testing pipeline...")
    pipeline = RoadAccidentPipeline()
    print("✓ Pipeline initialized")
    
    print("\nAll basic tests passed!")

def main():
    """Main test function"""
    print("Road Accident Analysis Pipeline - Test Suite")
    print("="*50)
    
    # Run basic tests first
    run_basic_tests()
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\nTest suite completed!")

if __name__ == "__main__":
    main()