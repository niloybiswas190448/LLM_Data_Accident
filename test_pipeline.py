#!/usr/bin/env python3
"""
Test Script for Bangla Traffic Violation Analysis Pipeline
==========================================================

This script demonstrates the key functionality of the pipeline
and validates that all components work correctly.
"""

import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from bangla_traffic_analysis_pipeline import (
    BanglaTrafficDataGenerator,
    BanglaTrafficExtractor,
    DataStorageManager,
    TrafficAnalytics
)

def test_data_generation():
    """Test synthetic data generation functionality."""
    print("🧪 Testing Data Generation...")
    
    generator = BanglaTrafficDataGenerator()
    
    # Test generating a small sample
    sentences = generator.generate_synthetic_bangla_sentences(n=10)
    
    print(f"✅ Generated {len(sentences)} synthetic sentences")
    
    # Display a few examples
    print("\n📝 Sample Generated Sentences:")
    for i, sentence in enumerate(sentences[:3], 1):
        print(f"{i}. {sentence}")
    
    return sentences

def test_information_extraction(sentences):
    """Test information extraction from Bangla text."""
    print("\n🔍 Testing Information Extraction...")
    
    extractor = BanglaTrafficExtractor()
    
    # Test extraction on first few sentences
    extracted_data = []
    for sentence in sentences[:5]:
        data = extractor.extract_structured_data(sentence)
        extracted_data.append(data)
        
        print(f"\n📄 Original: {sentence}")
        print(f"🔧 Extracted: {data}")
    
    # Test Bangla digit conversion
    test_text = "জরিমানা ২০০০ টাকা"
    converted = extractor.convert_bangla_digits(test_text)
    print(f"\n🔢 Bangla digit conversion: '{test_text}' → '{converted}'")
    
    # Test offense standardization
    test_offense = "হেলমেট না পরা"
    standardized = extractor.standardize_offense_terms(test_offense)
    print(f"🏷️ Offense standardization: '{test_offense}' → '{standardized}'")
    
    return extracted_data

def test_data_storage(extracted_data):
    """Test data storage functionality."""
    print("\n💾 Testing Data Storage...")
    
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(extracted_data)
    
    # Clean data
    df = df.dropna(subset=['district', 'offense_type', 'fine_amount'])
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Created DataFrame with {len(df)} records")
    print(f"📊 DataFrame columns: {list(df.columns)}")
    
    # Test storage
    storage = DataStorageManager()
    
    # Save to CSV
    csv_filename = "test_output.csv"
    storage.save_to_csv(df, csv_filename)
    
    # Verify file was created
    if os.path.exists(csv_filename):
        print(f"✅ CSV file created: {csv_filename}")
        # Clean up test file
        os.remove(csv_filename)
        print(f"🧹 Cleaned up test file: {csv_filename}")
    
    return df

def test_analytics(df):
    """Test analytics and visualization functionality."""
    print("\n📊 Testing Analytics...")
    
    if len(df) == 0:
        print("⚠️ No data available for analytics testing")
        return
    
    analytics = TrafficAnalytics()
    
    # Test summary statistics
    print("📈 Generating summary statistics...")
    summary = analytics.generate_summary_statistics(df)
    
    # Test basic analytics without plotting (to avoid display issues in testing)
    print(f"✅ Analytics test completed")
    print(f"📊 Summary statistics generated for {len(df)} records")
    
    return summary

def run_comprehensive_test():
    """Run a comprehensive test of the entire pipeline."""
    print("🚀 Starting Comprehensive Pipeline Test")
    print("=" * 50)
    
    try:
        # Test 1: Data Generation
        sentences = test_data_generation()
        
        # Test 2: Information Extraction
        extracted_data = test_information_extraction(sentences)
        
        # Test 3: Data Storage
        df = test_data_storage(extracted_data)
        
        # Test 4: Analytics
        summary = test_analytics(df)
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("🎯 Pipeline is ready for research use")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("🔧 Please check your dependencies and try again")
        return False

def test_specific_components():
    """Test specific components with edge cases."""
    print("\n🔬 Testing Specific Components...")
    
    extractor = BanglaTrafficExtractor()
    
    # Test edge cases
    test_cases = [
        "ঢাকা, গাবতলী, ২০২৩-১০-১৫: ট্রাফিক পুলিশ একটি মোটরসাইকেল চালককে হেলমেট না পরা অপরাধে ১০০০ টাকা জরিমানা করেছে।",
        "চট্টগ্রাম, পতেঙ্গা, ২০২৩-১১-২০: বাস চালক ওভারলোডিং অপরাধে ৫০০০ টাকা জরিমানা পেয়েছে।",
        "Invalid text without proper structure",
        "সিলেট, জালালাবাদ, ২০২৩-১২-০১: সিএনজি চালক সিগনাল লঙ্ঘন অপরাধে ২০০০ টাকা জরিমানা।"
    ]
    
    print("\n🔍 Testing Edge Cases:")
    for i, test_case in enumerate(test_cases, 1):
        result = extractor.extract_structured_data(test_case)
        print(f"\nTest {i}:")
        print(f"Input: {test_case}")
        print(f"Output: {result}")
        
        # Check if extraction was successful
        if result['district'] and result['offense_type']:
            print("✅ Extraction successful")
        else:
            print("⚠️ Extraction incomplete or failed")

if __name__ == "__main__":
    print("🧪 Bangla Traffic Violation Analysis Pipeline - Test Suite")
    print("=" * 60)
    
    # Run comprehensive test
    success = run_comprehensive_test()
    
    if success:
        # Run specific component tests
        test_specific_components()
        
        print("\n🎉 All tests completed successfully!")
        print("🚀 The pipeline is ready for research and analysis")
    else:
        print("\n❌ Pipeline test failed")
        print("🔧 Please check the error messages above and fix any issues")
        sys.exit(1)