#!/usr/bin/env python3
"""
Example Usage Script for Road Accident Analysis Pipeline
Demonstrates various ways to use the pipeline programmatically
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import RoadAccidentPipeline
from utils.logger import get_pipeline_logger

def example_basic_usage():
    """Example 1: Basic pipeline usage"""
    print("="*60)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RoadAccidentPipeline()
    
    # Run complete pipeline
    results = pipeline.run_full_pipeline()
    
    # Print results
    print(f"Pipeline completed: {results['success']}")
    print(f"Articles scraped: {results['scraped_articles']}")
    print(f"Records extracted: {results['extracted_records']}")
    print(f"Records stored: {results['stored_records']}")
    print(f"Visualizations created: {len(results['visualization_files'])}")
    
    if results['analysis_report']:
        stats = results['analysis_report']['summary_stats']
        print(f"\nSummary Statistics:")
        print(f"  Total Accidents: {stats['total_accidents']}")
        print(f"  Total Fatalities: {stats['total_fatalities']}")
        print(f"  Total Injuries: {stats['total_injuries']}")

def example_custom_date_range():
    """Example 2: Analysis for custom date range"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Date Range Analysis")
    print("="*60)
    
    pipeline = RoadAccidentPipeline()
    
    # Define custom date range (last 3 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Generate report for custom period
    report = pipeline.analyzer.generate_summary_report(start_date, end_date)
    
    print(f"Analysis Period: {start_date.date()} to {end_date.date()}")
    print(f"Total Days: {report['period']['total_days']}")
    
    stats = report['summary_stats']
    print(f"\nStatistics for the period:")
    print(f"  Total Accidents: {stats['total_accidents']}")
    print(f"  Fatal Accidents: {stats['fatal_accidents']}")
    print(f"  Major Accidents: {stats['major_accidents']}")
    print(f"  Minor Accidents: {stats['minor_accidents']}")
    
    # Show insights
    if report['insights']:
        insights = report['insights']
        if 'most_dangerous_district' in insights:
            district = insights['most_dangerous_district']
            print(f"\nMost Dangerous District: {district['name']}")
            print(f"  Accidents: {district['accidents']}")
            print(f"  Fatalities: {district['fatalities']}")
        
        if 'trend' in insights:
            print(f"\nTrend: {insights['trend']} ({insights['trend_percentage']:.1f}%)")

def example_monthly_report():
    """Example 3: Generate monthly report"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Monthly Report Generation")
    print("="*60)
    
    pipeline = RoadAccidentPipeline()
    
    # Generate report for last month
    current_date = datetime.now()
    last_month = current_date.replace(day=1) - timedelta(days=1)
    year = last_month.year
    month = last_month.month
    
    print(f"Generating monthly report for {year}-{month:02d}")
    
    results = pipeline.run_monthly_report(year, month)
    
    if results['success']:
        print("Monthly report generated successfully!")
        print(f"Visualizations created: {len(results['visualizations'])}")
        print(f"CSV export: {results['csv_export']}")
        
        # Show report summary
        report = results['report']
        stats = report['summary_stats']
        print(f"\nMonthly Summary:")
        print(f"  Accidents: {stats['total_accidents']}")
        print(f"  Fatalities: {stats['total_fatalities']}")
        print(f"  Injuries: {stats['total_injuries']}")
    else:
        print(f"Report generation failed: {results['error']}")

def example_data_retrieval():
    """Example 4: Retrieve and analyze specific data"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Data Retrieval and Analysis")
    print("="*60)
    
    pipeline = RoadAccidentPipeline()
    
    # Get recent accidents (last 30 days)
    recent_data = pipeline.data_manager.get_accidents(
        start_date=datetime.now() - timedelta(days=30),
        limit=10
    )
    
    print(f"Recent accidents (last 30 days): {len(recent_data)}")
    
    if not recent_data.empty:
        print("\nRecent Accident Details:")
        for idx, row in recent_data.head(5).iterrows():
            print(f"  {row['date'].date()}: {row['district']} - {row['fatalities']} fatalities")
    
    # Get district statistics
    district_stats = pipeline.data_manager.get_district_stats()
    
    if not district_stats.empty:
        print(f"\nTop 5 Districts by Accident Count:")
        for district, stats in district_stats.head(5).iterrows():
            print(f"  {district}: {stats['total_accidents']} accidents, {stats['fatalities']} fatalities")
    
    # Get monthly trends
    monthly_trends = pipeline.data_manager.get_monthly_trends()
    
    if not monthly_trends.empty:
        print(f"\nRecent Monthly Trends:")
        for period, stats in monthly_trends.tail(3).iterrows():
            print(f"  {period}: {stats['total_accidents']} accidents, {stats['fatalities']} fatalities")

def example_custom_analysis():
    """Example 5: Custom analysis scenarios"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Analysis Scenarios")
    print("="*60)
    
    pipeline = RoadAccidentPipeline()
    
    # Scenario 1: Fatal accidents only
    fatal_accidents = pipeline.data_manager.get_accidents(severity='fatal')
    print(f"Total fatal accidents: {len(fatal_accidents)}")
    
    if not fatal_accidents.empty:
        avg_fatalities = fatal_accidents['fatalities'].mean()
        print(f"Average fatalities per fatal accident: {avg_fatalities:.1f}")
    
    # Scenario 2: Accidents in specific district
    dhaka_accidents = pipeline.data_manager.get_accidents(district='Dhaka')
    print(f"Accidents in Dhaka: {len(dhaka_accidents)}")
    
    if not dhaka_accidents.empty:
        dhaka_stats = pipeline.data_manager.get_summary_stats()
        print(f"Dhaka accident statistics:")
        print(f"  Total: {len(dhaka_accidents)}")
        print(f"  Fatalities: {dhaka_accidents['fatalities'].sum()}")
        print(f"  Injuries: {dhaka_accidents['injuries'].sum()}")
    
    # Scenario 3: Vehicle type analysis
    all_accidents = pipeline.data_manager.get_accidents()
    if not all_accidents.empty:
        vehicle_types = all_accidents['vehicle_types'].value_counts()
        print(f"\nVehicle Types Involved:")
        for vehicle_type, count in vehicle_types.head(5).items():
            if vehicle_type:
                print(f"  {vehicle_type}: {count} accidents")

def example_system_management():
    """Example 6: System management operations"""
    print("\n" + "="*60)
    print("EXAMPLE 6: System Management")
    print("="*60)
    
    pipeline = RoadAccidentPipeline()
    
    # Check system status
    status = pipeline.get_system_status()
    print("System Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Export data to CSV
    csv_file = pipeline.data_manager.export_to_csv()
    if csv_file:
        print(f"\nData exported to: {csv_file}")
    
    # Create database backup
    backup_path = pipeline.backup_database()
    if backup_path:
        print(f"Database backed up to: {backup_path}")

def example_error_handling():
    """Example 7: Error handling demonstration"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Error Handling")
    print("="*60)
    
    pipeline = RoadAccidentPipeline()
    
    try:
        # Try to get data for a very old date range (should be empty)
        old_data = pipeline.data_manager.get_accidents(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 31)
        )
        
        if old_data.empty:
            print("No data found for the specified period (expected)")
        else:
            print(f"Found {len(old_data)} records for old period")
            
    except Exception as e:
        print(f"Error handled gracefully: {str(e)}")
    
    # Test with invalid parameters
    try:
        invalid_data = pipeline.data_manager.get_accidents(district='NonExistentDistrict')
        print(f"Invalid district query returned {len(invalid_data)} records")
    except Exception as e:
        print(f"Error with invalid parameters: {str(e)}")

def main():
    """Run all examples"""
    print("Road Accident Analysis Pipeline - Example Usage")
    print("This script demonstrates various ways to use the pipeline")
    
    # Run examples
    example_basic_usage()
    example_custom_date_range()
    example_monthly_report()
    example_data_retrieval()
    example_custom_analysis()
    example_system_management()
    example_error_handling()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nTo run the pipeline manually, use:")
    print("  python main.py")
    print("\nFor automated scheduling:")
    print("  python main.py --mode schedule")
    print("\nFor more options:")
    print("  python main.py --help")

if __name__ == "__main__":
    main()