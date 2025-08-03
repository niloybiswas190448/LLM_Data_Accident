#!/usr/bin/env python3
"""
Main Pipeline Script for Road Accident Analysis
Automated system for extracting, analyzing, and visualizing road accident data from Bangladeshi news sources
"""

import sys
import os
import schedule
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrapers.news_scraper import NewsScraper
from nlp.information_extractor import InformationExtractor
from database.data_manager import DataManager
from analysis.trend_analyzer import TrendAnalyzer
from utils.logger import get_pipeline_logger, log_execution_time
from config import START_DATE, END_DATE, ANALYSIS_CONFIG

class RoadAccidentPipeline:
    """
    Main pipeline class that orchestrates the entire road accident analysis system
    """
    
    def __init__(self):
        self.logger = get_pipeline_logger()
        self.scraper = NewsScraper()
        self.extractor = InformationExtractor()
        self.data_manager = DataManager()
        self.analyzer = TrendAnalyzer(self.data_manager)
        
        self.logger.info("Road Accident Analysis Pipeline initialized")
    
    @log_execution_time
    def run_full_pipeline(self, 
                         scrape_new_data: bool = True,
                         generate_analysis: bool = True,
                         create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline from scraping to analysis
        
        Args:
            scrape_new_data: Whether to scrape new data
            generate_analysis: Whether to generate analysis reports
            create_visualizations: Whether to create visualizations
            
        Returns:
            Dictionary containing pipeline results
        """
        pipeline_results = {
            'start_time': datetime.now(),
            'scraped_articles': 0,
            'extracted_records': 0,
            'stored_records': 0,
            'analysis_report': None,
            'visualization_files': [],
            'errors': []
        }
        
        try:
            # Step 1: Scrape news articles
            if scrape_new_data:
                self.logger.info("Starting news scraping phase...")
                articles = self.scraper.scrape_all_sources()
                pipeline_results['scraped_articles'] = len(articles)
                self.logger.info(f"Scraped {len(articles)} articles")
                
                if articles:
                    # Step 2: Extract structured information
                    self.logger.info("Starting information extraction phase...")
                    extracted_data = self.extractor.extract_batch(articles)
                    pipeline_results['extracted_records'] = len(extracted_data)
                    self.logger.info(f"Extracted information from {len(extracted_data)} articles")
                    
                    # Step 3: Store data
                    if extracted_data:
                        self.logger.info("Starting data storage phase...")
                        stored_count = self.data_manager.insert_batch(extracted_data)
                        pipeline_results['stored_records'] = stored_count
                        self.logger.info(f"Stored {stored_count} records in database")
            
            # Step 4: Generate analysis
            if generate_analysis:
                self.logger.info("Starting analysis phase...")
                analysis_report = self.analyzer.generate_summary_report()
                pipeline_results['analysis_report'] = analysis_report
                self.logger.info("Analysis report generated")
            
            # Step 5: Create visualizations
            if create_visualizations:
                self.logger.info("Starting visualization phase...")
                viz_files = self.analyzer.create_visualizations()
                pipeline_results['visualization_files'] = viz_files
                self.logger.info(f"Generated {len(viz_files)} visualization files")
            
            # Step 6: Export data to CSV
            csv_file = self.data_manager.export_to_csv()
            if csv_file:
                pipeline_results['csv_export'] = csv_file
                self.logger.info(f"Data exported to {csv_file}")
            
            pipeline_results['end_time'] = datetime.now()
            pipeline_results['success'] = True
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            pipeline_results['errors'].append(str(e))
            pipeline_results['success'] = False
            pipeline_results['end_time'] = datetime.now()
            self.logger.error(f"Pipeline failed: {str(e)}")
        
        return pipeline_results
    
    def run_weekly_update(self):
        """Run weekly automated update"""
        self.logger.info("Starting weekly automated update...")
        results = self.run_full_pipeline(scrape_new_data=True, 
                                       generate_analysis=True, 
                                       create_visualizations=True)
        
        # Log results
        if results['success']:
            self.logger.info(f"Weekly update completed: {results['stored_records']} new records")
        else:
            self.logger.error("Weekly update failed")
        
        return results
    
    def run_monthly_report(self, year: int, month: int):
        """Generate monthly report for specific month"""
        self.logger.info(f"Generating monthly report for {year}-{month:02d}")
        
        try:
            # Generate report
            report = self.analyzer.generate_monthly_report(year, month)
            
            # Create visualizations
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
            
            viz_files = self.analyzer.create_visualizations(start_date, end_date)
            
            # Export monthly data
            csv_file = self.data_manager.export_to_csv(start_date, end_date)
            
            monthly_results = {
                'report': report,
                'visualizations': viz_files,
                'csv_export': csv_file,
                'success': True
            }
            
            self.logger.info(f"Monthly report for {year}-{month:02d} completed")
            return monthly_results
            
        except Exception as e:
            self.logger.error(f"Monthly report generation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        try:
            # Get recent data statistics
            recent_data = self.data_manager.get_accidents(
                start_date=datetime.now() - timedelta(days=30)
            )
            
            # Get overall statistics
            overall_stats = self.data_manager.get_summary_stats()
            
            # Get database info
            db_size = os.path.getsize(self.data_manager.db_path) if os.path.exists(self.data_manager.db_path) else 0
            
            status = {
                'system_status': 'operational',
                'last_update': datetime.now(),
                'recent_records_30_days': len(recent_data),
                'total_records': overall_stats.get('total_accidents', 0),
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'output_directory': ANALYSIS_CONFIG["output_directory"],
                'database_path': self.data_manager.db_path
            }
            
            return status
            
        except Exception as e:
            return {
                'system_status': 'error',
                'error': str(e),
                'last_update': datetime.now()
            }
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data from database"""
        self.logger.info(f"Cleaning up data older than {days_to_keep} days...")
        deleted_count = self.data_manager.cleanup_old_data(days_to_keep)
        self.logger.info(f"Deleted {deleted_count} old records")
        return deleted_count
    
    def backup_database(self):
        """Create database backup"""
        self.logger.info("Creating database backup...")
        backup_path = self.data_manager.backup_database()
        if backup_path:
            self.logger.info(f"Database backed up to {backup_path}")
        return backup_path

def setup_scheduler(pipeline: RoadAccidentPipeline):
    """Set up automated scheduling"""
    # Weekly update every Sunday at 2 AM
    schedule.every().sunday.at("02:00").do(pipeline.run_weekly_update)
    
    # Monthly cleanup on 1st of each month at 3 AM
    schedule.every().month.at("03:00").do(pipeline.cleanup_old_data)
    
    # Monthly backup on 15th of each month at 4 AM
    schedule.every().month.at("04:00").do(pipeline.backup_database)
    
    pipeline.logger.info("Scheduler set up successfully")

def run_scheduler(pipeline: RoadAccidentPipeline):
    """Run the scheduler loop"""
    pipeline.logger.info("Starting scheduler...")
    setup_scheduler(pipeline)
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            pipeline.logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            pipeline.logger.error(f"Scheduler error: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Road Accident Analysis Pipeline')
    parser.add_argument('--mode', choices=['run', 'schedule', 'report', 'status', 'cleanup', 'backup'],
                       default='run', help='Pipeline mode')
    parser.add_argument('--year', type=int, help='Year for monthly report')
    parser.add_argument('--month', type=int, help='Month for monthly report')
    parser.add_argument('--no-scrape', action='store_true', help='Skip scraping phase')
    parser.add_argument('--no-analysis', action='store_true', help='Skip analysis phase')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization phase')
    parser.add_argument('--days-keep', type=int, default=365, help='Days to keep for cleanup')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RoadAccidentPipeline()
    
    try:
        if args.mode == 'run':
            # Run full pipeline
            results = pipeline.run_full_pipeline(
                scrape_new_data=not args.no_scrape,
                generate_analysis=not args.no_analysis,
                create_visualizations=not args.no_viz
            )
            
            # Print results
            print("\n" + "="*50)
            print("PIPELINE RESULTS")
            print("="*50)
            print(f"Success: {results['success']}")
            print(f"Scraped Articles: {results['scraped_articles']}")
            print(f"Extracted Records: {results['extracted_records']}")
            print(f"Stored Records: {results['stored_records']}")
            print(f"Visualization Files: {len(results['visualization_files'])}")
            
            if results['analysis_report']:
                stats = results['analysis_report']['summary_stats']
                print(f"\nSUMMARY STATISTICS:")
                print(f"Total Accidents: {stats['total_accidents']}")
                print(f"Total Fatalities: {stats['total_fatalities']}")
                print(f"Total Injuries: {stats['total_injuries']}")
                print(f"Fatal Accidents: {stats['fatal_accidents']}")
                print(f"Major Accidents: {stats['major_accidents']}")
                print(f"Minor Accidents: {stats['minor_accidents']}")
            
            if results['errors']:
                print(f"\nERRORS: {results['errors']}")
            
        elif args.mode == 'schedule':
            # Run scheduler
            run_scheduler(pipeline)
            
        elif args.mode == 'report':
            # Generate monthly report
            if not args.year or not args.month:
                print("Error: --year and --month are required for report mode")
                return
            
            results = pipeline.run_monthly_report(args.year, args.month)
            if results['success']:
                print(f"Monthly report for {args.year}-{args.month:02d} generated successfully")
                print(f"Visualizations: {len(results['visualizations'])}")
                print(f"CSV Export: {results['csv_export']}")
            else:
                print(f"Report generation failed: {results['error']}")
                
        elif args.mode == 'status':
            # Show system status
            status = pipeline.get_system_status()
            print("\n" + "="*50)
            print("SYSTEM STATUS")
            print("="*50)
            for key, value in status.items():
                print(f"{key}: {value}")
                
        elif args.mode == 'cleanup':
            # Clean up old data
            deleted_count = pipeline.cleanup_old_data(args.days_keep)
            print(f"Cleaned up {deleted_count} old records")
            
        elif args.mode == 'backup':
            # Create backup
            backup_path = pipeline.backup_database()
            if backup_path:
                print(f"Database backed up to: {backup_path}")
            else:
                print("Backup failed")
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"Pipeline error: {str(e)}")
        pipeline.logger.error(f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    main()