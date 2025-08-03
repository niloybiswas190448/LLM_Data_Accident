"""
Trend Analyzer Module for Road Accident Analysis Pipeline
Generates comprehensive analysis and visualizations of accident data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import folium
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

from config import ANALYSIS_CONFIG, START_DATE, END_DATE
from utils.logger import get_pipeline_logger, log_execution_time

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrendAnalyzer:
    """
    Analyzes accident data trends and generates visualizations
    """
    
    def __init__(self, data_manager):
        self.logger = get_pipeline_logger()
        self.data_manager = data_manager
        self.output_dir = ANALYSIS_CONFIG["output_directory"]
        
    def generate_summary_report(self, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive summary report
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary containing summary statistics and insights
        """
        if start_date is None:
            start_date = START_DATE
        if end_date is None:
            end_date = END_DATE
        
        self.logger.info(f"Generating summary report from {start_date} to {end_date}")
        
        # Get summary statistics
        summary_stats = self.data_manager.get_summary_stats(start_date, end_date)
        
        # Get district statistics
        district_stats = self.data_manager.get_district_stats(start_date, end_date)
        
        # Get monthly trends
        monthly_trends = self.data_manager.get_monthly_trends(start_date, end_date)
        
        # Calculate additional insights
        insights = self._calculate_insights(summary_stats, district_stats, monthly_trends)
        
        report = {
            'period': {
                'start_date': start_date,
                'end_date': end_date,
                'total_days': (end_date - start_date).days
            },
            'summary_stats': summary_stats,
            'top_districts': district_stats.head(10).to_dict('index') if not district_stats.empty else {},
            'monthly_trends': monthly_trends.to_dict('index') if not monthly_trends.empty else {},
            'insights': insights
        }
        
        return report
    
    def _calculate_insights(self, 
                           summary_stats: Dict[str, Any],
                           district_stats: pd.DataFrame,
                           monthly_trends: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate insights from the data
        
        Args:
            summary_stats: Summary statistics
            district_stats: District-wise statistics
            monthly_trends: Monthly trend data
            
        Returns:
            Dictionary containing insights
        """
        insights = {}
        
        # Most dangerous district
        if not district_stats.empty:
            most_dangerous = district_stats.iloc[0]
            insights['most_dangerous_district'] = {
                'name': district_stats.index[0],
                'accidents': int(most_dangerous['total_accidents']),
                'fatalities': int(most_dangerous['fatalities'])
            }
        
        # Trend analysis
        if not monthly_trends.empty and len(monthly_trends) > 1:
            recent_trend = monthly_trends.tail(3)['total_accidents'].mean()
            older_trend = monthly_trends.head(3)['total_accidents'].mean()
            
            if recent_trend > older_trend:
                insights['trend'] = "increasing"
                insights['trend_percentage'] = ((recent_trend - older_trend) / older_trend) * 100
            else:
                insights['trend'] = "decreasing"
                insights['trend_percentage'] = ((older_trend - recent_trend) / older_trend) * 100
        
        # Severity distribution
        if summary_stats['total_accidents'] > 0:
            insights['severity_distribution'] = {
                'fatal_percentage': (summary_stats['fatal_accidents'] / summary_stats['total_accidents']) * 100,
                'major_percentage': (summary_stats['major_accidents'] / summary_stats['total_accidents']) * 100,
                'minor_percentage': (summary_stats['minor_accidents'] / summary_stats['total_accidents']) * 100
            }
        
        return insights
    
    @log_execution_time
    def create_visualizations(self, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> List[str]:
        """
        Create comprehensive visualizations
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of generated visualization file paths
        """
        if start_date is None:
            start_date = START_DATE
        if end_date is None:
            end_date = END_DATE
        
        self.logger.info("Creating visualizations...")
        
        generated_files = []
        
        try:
            # 1. Monthly trends plot
            monthly_file = self._create_monthly_trends_plot(start_date, end_date)
            if monthly_file:
                generated_files.append(monthly_file)
            
            # 2. District-wise accident heatmap
            district_file = self._create_district_heatmap(start_date, end_date)
            if district_file:
                generated_files.append(district_file)
            
            # 3. Severity distribution pie chart
            severity_file = self._create_severity_distribution_plot(start_date, end_date)
            if severity_file:
                generated_files.append(severity_file)
            
            # 4. Fatalities vs Injuries scatter plot
            scatter_file = self._create_fatalities_injuries_scatter(start_date, end_date)
            if scatter_file:
                generated_files.append(scatter_file)
            
            # 5. Interactive map
            map_file = self._create_interactive_map(start_date, end_date)
            if map_file:
                generated_files.append(map_file)
            
            # 6. Word cloud of accident descriptions
            wordcloud_file = self._create_wordcloud(start_date, end_date)
            if wordcloud_file:
                generated_files.append(wordcloud_file)
            
            self.logger.info(f"Generated {len(generated_files)} visualization files")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
        
        return generated_files
    
    def _create_monthly_trends_plot(self, start_date: datetime, end_date: datetime) -> Optional[str]:
        """Create monthly trends visualization"""
        try:
            monthly_trends = self.data_manager.get_monthly_trends(start_date, end_date)
            
            if monthly_trends.empty:
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Total accidents over time
            monthly_trends['total_accidents'].plot(kind='line', ax=ax1, marker='o')
            ax1.set_title('Monthly Accident Trends', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Accidents')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Fatalities over time
            monthly_trends['fatalities'].plot(kind='line', ax=ax2, marker='s', color='red')
            ax2.set_title('Monthly Fatalities', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Fatalities')
            ax2.set_xlabel('Month')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = f"monthly_trends_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.png"
            filepath = f"{self.output_dir}/{filename}"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating monthly trends plot: {str(e)}")
            return None
    
    def _create_district_heatmap(self, start_date: datetime, end_date: datetime) -> Optional[str]:
        """Create district-wise accident heatmap"""
        try:
            district_stats = self.data_manager.get_district_stats(start_date, end_date)
            
            if district_stats.empty:
                return None
            
            # Prepare data for heatmap
            top_districts = district_stats.head(15)  # Top 15 districts
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Heatmap 1: Total accidents
            accidents_matrix = top_districts['total_accidents'].values.reshape(1, -1)
            sns.heatmap(accidents_matrix, 
                       annot=True, 
                       fmt='d',
                       xticklabels=top_districts.index,
                       yticklabels=['Total Accidents'],
                       ax=ax1,
                       cmap='Reds')
            ax1.set_title('Total Accidents by District', fontweight='bold')
            ax1.set_xlabel('District')
            
            # Heatmap 2: Fatalities
            fatalities_matrix = top_districts['fatalities'].values.reshape(1, -1)
            sns.heatmap(fatalities_matrix, 
                       annot=True, 
                       fmt='d',
                       xticklabels=top_districts.index,
                       yticklabels=['Fatalities'],
                       ax=ax2,
                       cmap='OrRd')
            ax2.set_title('Fatalities by District', fontweight='bold')
            ax2.set_xlabel('District')
            
            plt.tight_layout()
            
            filename = f"district_heatmap_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.png"
            filepath = f"{self.output_dir}/{filename}"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating district heatmap: {str(e)}")
            return None
    
    def _create_severity_distribution_plot(self, start_date: datetime, end_date: datetime) -> Optional[str]:
        """Create severity distribution pie chart"""
        try:
            summary_stats = self.data_manager.get_summary_stats(start_date, end_date)
            
            if summary_stats['total_accidents'] == 0:
                return None
            
            # Prepare data for pie chart
            severity_data = {
                'Fatal': summary_stats['fatal_accidents'],
                'Major': summary_stats['major_accidents'],
                'Minor': summary_stats['minor_accidents']
            }
            
            # Remove zero values
            severity_data = {k: v for k, v in severity_data.items() if v > 0}
            
            if not severity_data:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Pie chart
            colors = ['#ff4444', '#ffaa00', '#44ff44']
            wedges, texts, autotexts = ax1.pie(severity_data.values(), 
                                              labels=severity_data.keys(),
                                              autopct='%1.1f%%',
                                              colors=colors[:len(severity_data)],
                                              startangle=90)
            ax1.set_title('Accident Severity Distribution', fontweight='bold')
            
            # Bar chart
            ax2.bar(severity_data.keys(), severity_data.values(), color=colors[:len(severity_data)])
            ax2.set_title('Accident Count by Severity', fontweight='bold')
            ax2.set_ylabel('Number of Accidents')
            
            # Add value labels on bars
            for i, v in enumerate(severity_data.values()):
                ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            filename = f"severity_distribution_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.png"
            filepath = f"{self.output_dir}/{filename}"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating severity distribution plot: {str(e)}")
            return None
    
    def _create_fatalities_injuries_scatter(self, start_date: datetime, end_date: datetime) -> Optional[str]:
        """Create fatalities vs injuries scatter plot"""
        try:
            df = self.data_manager.get_accidents(start_date, end_date)
            
            if df.empty:
                return None
            
            # Filter out records with both fatalities and injuries
            df_filtered = df[(df['fatalities'] > 0) | (df['injuries'] > 0)]
            
            if df_filtered.empty:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot with size based on severity
            severity_colors = {'fatal': 'red', 'major': 'orange', 'minor': 'green'}
            severity_sizes = {'fatal': 100, 'major': 60, 'minor': 30}
            
            for severity in severity_colors.keys():
                subset = df_filtered[df_filtered['severity'] == severity]
                if not subset.empty:
                    ax.scatter(subset['fatalities'], subset['injuries'],
                              c=severity_colors[severity],
                              s=severity_sizes[severity],
                              alpha=0.7,
                              label=severity.capitalize())
            
            ax.set_xlabel('Number of Fatalities')
            ax.set_ylabel('Number of Injuries')
            ax.set_title('Fatalities vs Injuries by Accident Severity', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(df_filtered) > 1:
                z = np.polyfit(df_filtered['fatalities'], df_filtered['injuries'], 1)
                p = np.poly1d(z)
                ax.plot(df_filtered['fatalities'], p(df_filtered['fatalities']), "r--", alpha=0.8)
            
            plt.tight_layout()
            
            filename = f"fatalities_injuries_scatter_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.png"
            filepath = f"{self.output_dir}/{filename}"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating fatalities-injuries scatter plot: {str(e)}")
            return None
    
    def _create_interactive_map(self, start_date: datetime, end_date: datetime) -> Optional[str]:
        """Create interactive map of accident locations"""
        try:
            df = self.data_manager.get_accidents(start_date, end_date)
            
            if df.empty:
                return None
            
            # Filter records with location information
            df_with_location = df[df['district'].notna() & (df['district'] != '')]
            
            if df_with_location.empty:
                return None
            
            # Create map centered on Bangladesh
            m = folium.Map(location=[23.6850, 90.3563], zoom_start=7)
            
            # Add accident markers
            for _, row in df_with_location.iterrows():
                # Use district coordinates (simplified - in real implementation, you'd have a coordinate database)
                district_coords = self._get_district_coordinates(row['district'])
                
                if district_coords:
                    # Determine marker color based on severity
                    color_map = {'fatal': 'red', 'major': 'orange', 'minor': 'green'}
                    color = color_map.get(row['severity'], 'blue')
                    
                    # Create popup content
                    popup_content = f"""
                    <b>Accident Details</b><br>
                    Date: {row['date']}<br>
                    District: {row['district']}<br>
                    Fatalities: {row['fatalities']}<br>
                    Injuries: {row['injuries']}<br>
                    Severity: {row['severity']}<br>
                    Source: {row['source_name']}
                    """
                    
                    folium.Marker(
                        location=district_coords,
                        popup=folium.Popup(popup_content, max_width=300),
                        icon=folium.Icon(color=color, icon='car', prefix='fa')
                    ).add_to(m)
            
            filename = f"accident_map_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.html"
            filepath = f"{self.output_dir}/{filename}"
            m.save(filepath)
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating interactive map: {str(e)}")
            return None
    
    def _create_wordcloud(self, start_date: datetime, end_date: datetime) -> Optional[str]:
        """Create word cloud from accident descriptions"""
        try:
            df = self.data_manager.get_accidents(start_date, end_date)
            
            if df.empty:
                return None
            
            # Combine all text
            text = ' '.join(df['translated_text'].dropna().astype(str))
            
            if not text.strip():
                return None
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud of Accident Descriptions', fontweight='bold', fontsize=16)
            
            filename = f"wordcloud_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}.png"
            filepath = f"{self.output_dir}/{filename}"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating word cloud: {str(e)}")
            return None
    
    def _get_district_coordinates(self, district_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a district (simplified implementation)"""
        # Simplified coordinate mapping - in a real implementation, you'd have a complete database
        coordinates = {
            'Dhaka': (23.8103, 90.4125),
            'Chittagong': (22.3419, 91.8132),
            'Rajshahi': (24.3745, 88.6042),
            'Khulna': (22.8456, 89.5403),
            'Barisal': (22.7010, 90.3535),
            'Sylhet': (24.8949, 91.8687),
            'Rangpur': (25.7439, 89.2752),
            'Mymensingh': (24.7471, 90.4203)
        }
        
        return coordinates.get(district_name)
    
    def generate_monthly_report(self, year: int, month: int) -> Dict[str, Any]:
        """
        Generate monthly report for a specific month
        
        Args:
            year: Year
            month: Month (1-12)
            
        Returns:
            Monthly report dictionary
        """
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        return self.generate_summary_report(start_date, end_date)