#!/usr/bin/env python3
"""
Example Usage: Bangla Traffic Violation Analysis Pipeline
========================================================

This script demonstrates how to use the pipeline for specific research scenarios
and shows various ways to customize and extend the functionality.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import pipeline components
from bangla_traffic_analysis_pipeline import (
    BanglaTrafficDataGenerator,
    BanglaTrafficExtractor,
    DataStorageManager,
    TrafficAnalytics
)

def example_1_basic_analysis():
    """
    Example 1: Basic Traffic Violation Analysis
    ===========================================
    
    This example shows how to run a basic analysis of traffic violations
    with default settings.
    """
    print("📊 Example 1: Basic Traffic Violation Analysis")
    print("=" * 50)
    
    # Initialize components
    generator = BanglaTrafficDataGenerator()
    extractor = BanglaTrafficExtractor()
    storage = DataStorageManager()
    analytics = TrafficAnalytics()
    
    # Generate data
    print("📝 Generating synthetic data...")
    sentences = generator.generate_synthetic_bangla_sentences(n=200)
    
    # Extract structured data
    print("🔍 Extracting structured information...")
    extracted_data = []
    for sentence in sentences:
        data = extractor.extract_structured_data(sentence)
        if data['district'] and data['offense_type']:
            data['offense_type'] = extractor.standardize_offense_terms(data['offense_type'])
            extracted_data.append(data)
    
    # Create DataFrame
    df = pd.DataFrame(extracted_data)
    df = df.dropna(subset=['district', 'offense_type', 'fine_amount'])
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Processed {len(df)} valid records")
    
    # Save data
    storage.save_output(df)
    
    # Generate analytics
    summary = analytics.generate_summary_statistics(df)
    
    # Create visualizations
    analytics.visualize_offense_distribution(df)
    analytics.visualize_monthly_trends(df)
    analytics.visualize_district_analysis(df)
    
    return df

def example_2_district_comparison():
    """
    Example 2: District-wise Comparison Analysis
    ===========================================
    
    This example focuses on comparing traffic violations across different districts.
    """
    print("\n🏛️ Example 2: District-wise Comparison Analysis")
    print("=" * 50)
    
    # Generate data with focus on specific districts
    generator = BanglaTrafficDataGenerator()
    extractor = BanglaTrafficExtractor()
    
    # Generate more data for better district comparison
    sentences = generator.generate_synthetic_bangla_sentences(n=300)
    
    # Extract and process data
    extracted_data = []
    for sentence in sentences:
        data = extractor.extract_structured_data(sentence)
        if data['district'] and data['offense_type']:
            data['offense_type'] = extractor.standardize_offense_terms(data['offense_type'])
            extracted_data.append(data)
    
    df = pd.DataFrame(extracted_data)
    df = df.dropna(subset=['district', 'offense_type', 'fine_amount'])
    df['date'] = pd.to_datetime(df['date'])
    
    # District-wise analysis
    print("\n📊 District-wise Statistics:")
    district_stats = df.groupby('district').agg({
        'fine_amount': ['count', 'sum', 'mean'],
        'offense_type': 'nunique'
    }).round(2)
    
    district_stats.columns = ['Violation_Count', 'Total_Fines', 'Average_Fine', 'Unique_Offense_Types']
    print(district_stats)
    
    # Find districts with highest violation rates
    top_districts = df['district'].value_counts().head(5)
    print(f"\n🏆 Top 5 Districts by Violation Count:")
    for district, count in top_districts.items():
        print(f"  {district}: {count} violations")
    
    # Analyze fine distribution by district
    print(f"\n💰 Fine Analysis by District:")
    for district in df['district'].unique():
        district_data = df[df['district'] == district]
        total_fines = district_data['fine_amount'].sum()
        avg_fine = district_data['fine_amount'].mean()
        print(f"  {district}: Total fines = {total_fines:,.0f} BDT, Average = {avg_fine:.0f} BDT")
    
    return df

def example_3_offense_pattern_analysis():
    """
    Example 3: Offense Pattern Analysis
    ===================================
    
    This example analyzes patterns in different types of traffic offenses.
    """
    print("\n🚨 Example 3: Offense Pattern Analysis")
    print("=" * 50)
    
    # Generate data
    generator = BanglaTrafficDataGenerator()
    extractor = BanglaTrafficExtractor()
    
    sentences = generator.generate_synthetic_bangla_sentences(n=250)
    
    # Extract and process data
    extracted_data = []
    for sentence in sentences:
        data = extractor.extract_structured_data(sentence)
        if data['district'] and data['offense_type']:
            data['offense_type'] = extractor.standardize_offense_terms(data['offense_type'])
            extracted_data.append(data)
    
    df = pd.DataFrame(extracted_data)
    df = df.dropna(subset=['district', 'offense_type', 'fine_amount'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Offense type analysis
    print("\n📋 Offense Type Analysis:")
    offense_stats = df.groupby('offense_type').agg({
        'fine_amount': ['count', 'sum', 'mean'],
        'district': 'nunique'
    }).round(2)
    
    offense_stats.columns = ['Occurrence_Count', 'Total_Fines', 'Average_Fine', 'Districts_Affected']
    print(offense_stats)
    
    # Most common offenses
    print(f"\n🔥 Most Common Offenses:")
    common_offenses = df['offense_type'].value_counts().head(5)
    for offense, count in common_offenses.items():
        percentage = (count / len(df)) * 100
        print(f"  {offense}: {count} times ({percentage:.1f}%)")
    
    # Fine analysis by offense type
    print(f"\n💰 Fine Analysis by Offense Type:")
    for offense in df['offense_type'].unique():
        offense_data = df[df['offense_type'] == offense]
        total_fines = offense_data['fine_amount'].sum()
        avg_fine = offense_data['fine_amount'].mean()
        count = len(offense_data)
        print(f"  {offense}: {count} violations, Total = {total_fines:,.0f} BDT, Avg = {avg_fine:.0f} BDT")
    
    # Vehicle type vs offense analysis
    print(f"\n🚗 Vehicle Type vs Offense Analysis:")
    vehicle_offense = df.groupby(['vehicle_type', 'offense_type']).size().unstack(fill_value=0)
    print(vehicle_offense)
    
    return df

def example_4_temporal_analysis():
    """
    Example 4: Temporal Pattern Analysis
    ====================================
    
    This example analyzes how traffic violations change over time.
    """
    print("\n📅 Example 4: Temporal Pattern Analysis")
    print("=" * 50)
    
    # Generate data
    generator = BanglaTrafficDataGenerator()
    extractor = BanglaTrafficExtractor()
    
    sentences = generator.generate_synthetic_bangla_sentences(n=400)
    
    # Extract and process data
    extracted_data = []
    for sentence in sentences:
        data = extractor.extract_structured_data(sentence)
        if data['district'] and data['offense_type']:
            data['offense_type'] = extractor.standardize_offense_terms(data['offense_type'])
            extracted_data.append(data)
    
    df = pd.DataFrame(extracted_data)
    df = df.dropna(subset=['district', 'offense_type', 'fine_amount'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Add temporal features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    df['quarter'] = df['date'].dt.quarter
    
    # Monthly analysis
    print("\n📊 Monthly Violation Patterns:")
    monthly_stats = df.groupby('month').agg({
        'fine_amount': ['count', 'sum', 'mean']
    }).round(2)
    
    monthly_stats.columns = ['Violation_Count', 'Total_Fines', 'Average_Fine']
    print(monthly_stats)
    
    # Day of week analysis
    print(f"\n📅 Day of Week Analysis:")
    dow_stats = df.groupby('day_of_week').agg({
        'fine_amount': ['count', 'sum', 'mean']
    }).round(2)
    
    dow_stats.columns = ['Violation_Count', 'Total_Fines', 'Average_Fine']
    print(dow_stats)
    
    # Quarterly analysis
    print(f"\n📈 Quarterly Analysis:")
    quarterly_stats = df.groupby('quarter').agg({
        'fine_amount': ['count', 'sum', 'mean']
    }).round(2)
    
    quarterly_stats.columns = ['Violation_Count', 'Total_Fines', 'Average_Fine']
    print(quarterly_stats)
    
    # Time series visualization
    analytics = TrafficAnalytics()
    analytics.visualize_monthly_trends(df)
    
    return df

def example_5_research_insights():
    """
    Example 5: Research Insights and Policy Recommendations
    ======================================================
    
    This example demonstrates how to extract research insights and generate
    policy recommendations from the analysis.
    """
    print("\n🎓 Example 5: Research Insights and Policy Recommendations")
    print("=" * 60)
    
    # Run comprehensive analysis
    df = example_1_basic_analysis()
    
    print("\n🔬 RESEARCH INSIGHTS:")
    print("-" * 30)
    
    # Insight 1: Most problematic offenses
    most_common_offense = df['offense_type'].mode().iloc[0]
    most_common_count = df[df['offense_type'] == most_common_offense].shape[0]
    print(f"1. Most Common Offense: {most_common_offense} ({most_common_count} cases)")
    
    # Insight 2: District with highest violations
    highest_violation_district = df['district'].mode().iloc[0]
    district_violations = df[df['district'] == highest_violation_district].shape[0]
    print(f"2. District with Most Violations: {highest_violation_district} ({district_violations} cases)")
    
    # Insight 3: Financial impact
    total_revenue = df['fine_amount'].sum()
    avg_fine = df['fine_amount'].mean()
    print(f"3. Total Fine Revenue: {total_revenue:,.0f} BDT")
    print(f"4. Average Fine Amount: {avg_fine:.0f} BDT")
    
    # Insight 4: Enforcement effectiveness
    unique_districts = df['district'].nunique()
    unique_offenses = df['offense_type'].nunique()
    print(f"5. Geographic Coverage: {unique_districts} districts")
    print(f"6. Offense Types Covered: {unique_offenses} categories")
    
    print("\n📋 POLICY RECOMMENDATIONS:")
    print("-" * 30)
    
    # Recommendation 1: Focus on most common offense
    print(f"1. Prioritize enforcement for '{most_common_offense}' violations")
    
    # Recommendation 2: District-specific strategies
    print(f"2. Develop targeted strategies for {highest_violation_district} district")
    
    # Recommendation 3: Fine optimization
    if avg_fine < 2000:
        print("3. Consider increasing fine amounts for better deterrent effect")
    else:
        print("3. Current fine levels appear appropriate")
    
    # Recommendation 4: Resource allocation
    print("4. Allocate enforcement resources based on violation frequency by district")
    
    # Recommendation 5: Monitoring and evaluation
    print("5. Establish regular monitoring system for violation trends")
    
    print("\n📊 RESEARCH METHODOLOGY:")
    print("-" * 30)
    print("• Data Source: Synthetic Bangla traffic violation reports")
    print("• Analysis Period: 12 months")
    print("• Geographic Coverage: 10 major districts")
    print("• Offense Categories: 11 violation types")
    print("• Statistical Methods: Descriptive analysis, trend analysis")
    print("• Visualization: Bar charts, time series, heatmaps")
    
    return df

def main():
    """
    Main function to run all examples.
    """
    print("🚀 Bangla Traffic Violation Analysis Pipeline - Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        print("\n" + "="*60)
        example_1_basic_analysis()
        
        print("\n" + "="*60)
        example_2_district_comparison()
        
        print("\n" + "="*60)
        example_3_offense_pattern_analysis()
        
        print("\n" + "="*60)
        example_4_temporal_analysis()
        
        print("\n" + "="*60)
        example_5_research_insights()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("🎯 Pipeline is ready for research and policy analysis")
        print("📚 Data and visualizations saved for further analysis")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("🔧 Please check your dependencies and try again")

if __name__ == "__main__":
    main()