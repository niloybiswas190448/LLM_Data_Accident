"""
Data Manager Module for Road Accident Analysis Pipeline
Handles data storage, retrieval, and management using SQLite and CSV
"""

import sqlite3
import pandas as pd
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
import os

from config import DATABASE_PATH, CSV_OUTPUT_PATH
from utils.logger import get_pipeline_logger, log_execution_time

class DataManager:
    """
    Manages data storage and retrieval for road accident analysis
    """
    
    def __init__(self):
        self.logger = get_pipeline_logger()
        self.db_path = DATABASE_PATH
        self.csv_path = CSV_OUTPUT_PATH
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create accidents table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS accidents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        title TEXT,
                        text TEXT,
                        translated_text TEXT,
                        district TEXT,
                        upazila TEXT,
                        road TEXT,
                        specific_location TEXT,
                        fatalities INTEGER DEFAULT 0,
                        injuries INTEGER DEFAULT 0,
                        vehicle_types TEXT,
                        severity TEXT,
                        source_url TEXT,
                        source_name TEXT,
                        language TEXT,
                        extraction_confidence REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON accidents(date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_district ON accidents(district)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON accidents(severity)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON accidents(source_name)')
                
                # Create summary statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS summary_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stat_date DATE,
                        total_accidents INTEGER DEFAULT 0,
                        total_fatalities INTEGER DEFAULT 0,
                        total_injuries INTEGER DEFAULT 0,
                        fatal_accidents INTEGER DEFAULT 0,
                        major_accidents INTEGER DEFAULT 0,
                        minor_accidents INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database setup completed successfully")
                
        except Exception as e:
            self.logger.error(f"Database setup failed: {str(e)}")
            raise
    
    def insert_accident(self, accident_data: Dict[str, Any]) -> bool:
        """
        Insert a single accident record into the database
        
        Args:
            accident_data: Dictionary containing accident information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert datetime to string if present
                date_str = None
                if accident_data.get('date'):
                    if isinstance(accident_data['date'], datetime):
                        date_str = accident_data['date'].isoformat()
                    else:
                        date_str = str(accident_data['date'])
                
                cursor.execute('''
                    INSERT INTO accidents (
                        date, title, text, translated_text, district, upazila, road,
                        specific_location, fatalities, injuries, vehicle_types, severity,
                        source_url, source_name, language, extraction_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date_str,
                    accident_data.get('title', ''),
                    accident_data.get('text', ''),
                    accident_data.get('translated_text', ''),
                    accident_data.get('district', ''),
                    accident_data.get('upazila', ''),
                    accident_data.get('road', ''),
                    accident_data.get('specific_location', ''),
                    accident_data.get('fatalities', 0),
                    accident_data.get('injuries', 0),
                    accident_data.get('vehicle_types', ''),
                    accident_data.get('severity', ''),
                    accident_data.get('source_url', ''),
                    accident_data.get('source_name', ''),
                    accident_data.get('language', ''),
                    accident_data.get('extraction_confidence', 0.0)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to insert accident data: {str(e)}")
            return False
    
    @log_execution_time
    def insert_batch(self, accidents_data: List[Dict[str, Any]]) -> int:
        """
        Insert multiple accident records into the database
        
        Args:
            accidents_data: List of accident data dictionaries
            
        Returns:
            Number of successfully inserted records
        """
        successful_inserts = 0
        
        for accident_data in accidents_data:
            if self.insert_accident(accident_data):
                successful_inserts += 1
        
        self.logger.info(f"Successfully inserted {successful_inserts}/{len(accidents_data)} accident records")
        return successful_inserts
    
    def get_accidents(self, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     district: Optional[str] = None,
                     severity: Optional[str] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve accident data from database with optional filters
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            district: District filter
            severity: Severity filter
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing accident data
        """
        try:
            query = "SELECT * FROM accidents WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.isoformat())
            
            if district:
                query += " AND district LIKE ?"
                params.append(f"%{district}%")
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            query += " ORDER BY date DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            # Convert date strings back to datetime
            if 'date' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve accident data: {str(e)}")
            return pd.DataFrame()
    
    def get_summary_stats(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get summary statistics for accidents
        
        Args:
            start_date: Start date for statistics
            end_date: End date for statistics
            
        Returns:
            Dictionary containing summary statistics
        """
        try:
            df = self.get_accidents(start_date, end_date)
            
            if df.empty:
                return {
                    'total_accidents': 0,
                    'total_fatalities': 0,
                    'total_injuries': 0,
                    'fatal_accidents': 0,
                    'major_accidents': 0,
                    'minor_accidents': 0,
                    'avg_fatalities_per_accident': 0,
                    'avg_injuries_per_accident': 0
                }
            
            stats = {
                'total_accidents': len(df),
                'total_fatalities': df['fatalities'].sum(),
                'total_injuries': df['injuries'].sum(),
                'fatal_accidents': len(df[df['severity'] == 'fatal']),
                'major_accidents': len(df[df['severity'] == 'major']),
                'minor_accidents': len(df[df['severity'] == 'minor']),
                'avg_fatalities_per_accident': df['fatalities'].mean(),
                'avg_injuries_per_accident': df['injuries'].mean()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get summary stats: {str(e)}")
            return {}
    
    def get_district_stats(self, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get accident statistics by district
        
        Args:
            start_date: Start date for statistics
            end_date: End date for statistics
            
        Returns:
            DataFrame with district-wise statistics
        """
        try:
            df = self.get_accidents(start_date, end_date)
            
            if df.empty:
                return pd.DataFrame()
            
            # Group by district and calculate statistics
            district_stats = df.groupby('district').agg({
                'id': 'count',
                'fatalities': 'sum',
                'injuries': 'sum',
                'severity': lambda x: (x == 'fatal').sum()
            }).rename(columns={
                'id': 'total_accidents',
                'severity': 'fatal_accidents'
            })
            
            district_stats['avg_fatalities'] = district_stats['fatalities'] / district_stats['total_accidents']
            district_stats['avg_injuries'] = district_stats['injuries'] / district_stats['total_accidents']
            
            return district_stats.sort_values('total_accidents', ascending=False)
            
        except Exception as e:
            self.logger.error(f"Failed to get district stats: {str(e)}")
            return pd.DataFrame()
    
    def get_monthly_trends(self, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get monthly accident trends
        
        Args:
            start_date: Start date for trends
            end_date: End date for trends
            
        Returns:
            DataFrame with monthly trends
        """
        try:
            df = self.get_accidents(start_date, end_date)
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert date to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Group by month
            df['year_month'] = df['date'].dt.to_period('M')
            
            monthly_trends = df.groupby('year_month').agg({
                'id': 'count',
                'fatalities': 'sum',
                'injuries': 'sum',
                'severity': lambda x: (x == 'fatal').sum()
            }).rename(columns={
                'id': 'total_accidents',
                'severity': 'fatal_accidents'
            })
            
            return monthly_trends.sort_index()
            
        except Exception as e:
            self.logger.error(f"Failed to get monthly trends: {str(e)}")
            return pd.DataFrame()
    
    def export_to_csv(self, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> str:
        """
        Export accident data to CSV file
        
        Args:
            start_date: Start date for export
            end_date: End date for export
            
        Returns:
            Path to the exported CSV file
        """
        try:
            df = self.get_accidents(start_date, end_date)
            
            if df.empty:
                self.logger.warning("No data to export")
                return ""
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accident_data_{timestamp}.csv"
            filepath = os.path.join(os.path.dirname(self.csv_path), filename)
            
            # Export to CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            self.logger.info(f"Data exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to export data: {str(e)}")
            return ""
    
    def backup_database(self) -> str:
        """
        Create a backup of the database
        
        Returns:
            Path to the backup file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"
            
            with sqlite3.connect(self.db_path) as source_conn:
                with sqlite3.connect(backup_path) as backup_conn:
                    source_conn.backup(backup_conn)
            
            self.logger.info(f"Database backed up to {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to backup database: {str(e)}")
            return ""
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """
        Remove old accident data from database
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM accidents WHERE date < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                conn.commit()
            
            self.logger.info(f"Deleted {deleted_count} old records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
            return 0