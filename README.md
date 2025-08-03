# Road Accident Analysis Pipeline for Bangladesh

A comprehensive, automated system for extracting, analyzing, and visualizing road accident data from Bangladeshi news sources using web scraping, Natural Language Processing (NLP), and Large Language Models (LLMs).

## 🎯 Project Overview

This pipeline automatically:
- **Scrapes** accident-related news from major Bangladeshi media outlets (Prothom Alo, Daily Star, Ittefaq, etc.)
- **Extracts** structured information using NLP and LLM techniques
- **Translates** Bangla content to English when needed
- **Stores** data in SQLite database and CSV formats
- **Analyzes** trends and patterns over time
- **Generates** comprehensive visualizations and reports
- **Runs** on automated schedules for continuous monitoring

## 🚀 Key Features

### 🔍 **Intelligent Data Extraction**
- Multi-language support (Bangla and English)
- Automatic translation using Google Translate API
- Named Entity Recognition for location extraction
- Rule-based and ML-based information extraction
- Vehicle type classification
- Severity level assessment

### 📊 **Comprehensive Analysis**
- Monthly/yearly trend analysis
- District-wise accident statistics
- Severity distribution analysis
- Fatalities vs injuries correlation
- Most dangerous roads identification
- Time-series forecasting capabilities

### 📈 **Rich Visualizations**
- Interactive maps with accident locations
- Monthly trend plots
- District-wise heatmaps
- Severity distribution charts
- Word clouds from accident descriptions
- Fatalities vs injuries scatter plots

### 🤖 **Full Automation**
- Scheduled weekly updates
- Automated data cleanup
- Database backups
- Error handling and recovery
- Comprehensive logging

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   News Sources  │───▶│   Web Scraper   │───▶│  Data Manager   │
│                 │    │                 │    │                 │
│ • Prothom Alo   │    │ • Rate Limiting │    │ • SQLite DB     │
│ • Daily Star    │    │ • Error Handling│    │ • CSV Export    │
│ • Ittefaq       │    │ • Multi-source  │    │ • Data Backup   │
│ • Jugantor      │    │ • Content Filter│    │ • Cleanup       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   NLP/LLM       │    │   Trend Analyzer│
                       │   Extractor     │    │                 │
                       │                 │    │ • Statistics    │
                       │ • Translation   │    │ • Visualizations│
                       │ • NER           │    │ • Reports       │
                       │ • Classification│    │ • Insights      │
                       │ • Rule-based    │    │ • Maps          │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection for web scraping
- Sufficient disk space for database and visualizations

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd road-accident-analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data:**
```python
python -c "import nltk; nltk.download('punkt')"
```

4. **Set up output directory:**
```bash
mkdir outputs
```

## 🚀 Quick Start

### Basic Usage

1. **Run the complete pipeline:**
```bash
python main.py
```

2. **Run with specific options:**
```bash
# Skip scraping (use existing data)
python main.py --no-scrape

# Skip analysis (only scrape and store)
python main.py --no-analysis

# Skip visualizations
python main.py --no-viz
```

### Advanced Usage

1. **Set up automated scheduling:**
```bash
python main.py --mode schedule
```

2. **Generate monthly report:**
```bash
python main.py --mode report --year 2024 --month 1
```

3. **Check system status:**
```bash
python main.py --mode status
```

4. **Clean up old data:**
```bash
python main.py --mode cleanup --days-keep 180
```

5. **Create database backup:**
```bash
python main.py --mode backup
```

## 📊 Output Files

The pipeline generates several types of output files in the `outputs/` directory:

### 📈 Visualizations
- `monthly_trends_YYYYMM_YYYYMM.png` - Monthly accident trends
- `district_heatmap_YYYYMM_YYYYMM.png` - District-wise accident heatmap
- `severity_distribution_YYYYMM_YYYYMM.png` - Severity distribution charts
- `fatalities_injuries_scatter_YYYYMM_YYYYMM.png` - Fatalities vs injuries scatter plot
- `wordcloud_YYYYMM_YYYYMM.png` - Word cloud from accident descriptions
- `accident_map_YYYYMM_YYYYMM.html` - Interactive map of accident locations

### 📄 Data Exports
- `accident_data_YYYYMMDD_HHMMSS.csv` - Structured accident data
- `road_accidents.db` - SQLite database with all data
- `road_accidents.db.backup_YYYYMMDD_HHMMSS` - Database backups

### 📋 Reports
- Console output with summary statistics
- Log files with detailed execution information

## ⚙️ Configuration

The system is highly configurable through `config.py`:

### News Sources
```python
NEWS_SOURCES = {
    "prothom_alo": {
        "base_url": "https://www.prothomalo.com",
        "search_url": "https://www.prothomalo.com/search?q={query}",
        "language": "bangla",
        # ... selectors and configuration
    }
}
```

### Keywords
```python
ACCIDENT_KEYWORDS = {
    "bangla": ["দুর্ঘটনা", "আঘাত", "মৃত্যু", "জখম", "রাস্তা"],
    "english": ["accident", "crash", "collision", "killed", "injured"]
}
```

### Analysis Settings
```python
ANALYSIS_CONFIG = {
    "trend_period": "monthly",
    "top_locations_count": 10,
    "visualization_format": "png",
    "output_directory": "outputs"
}
```

## 🔧 Customization

### Adding New News Sources

1. Add source configuration to `config.py`:
```python
"new_source": {
    "base_url": "https://example.com",
    "search_url": "https://example.com/search?q={query}",
    "language": "english",
    "selectors": {
        "article_links": ".article a",
        "title": "h1.title",
        "content": ".content",
        "date": ".published-date"
    }
}
```

2. Update the scraper to handle the new source format if needed.

### Modifying Extraction Rules

Edit the patterns in `nlp/information_extractor.py`:
- Number extraction patterns
- Date parsing patterns
- Location extraction rules
- Vehicle type classification

### Custom Visualizations

Add new visualization methods to `analysis/trend_analyzer.py`:
```python
def _create_custom_plot(self, start_date, end_date):
    # Your custom visualization code
    pass
```

## 📈 Sample Output

### Summary Statistics
```
PIPELINE RESULTS
==================================================
Success: True
Scraped Articles: 45
Extracted Records: 38
Stored Records: 38
Visualization Files: 6

SUMMARY STATISTICS:
Total Accidents: 156
Total Fatalities: 89
Total Injuries: 234
Fatal Accidents: 23
Major Accidents: 67
Minor Accidents: 66
```

### System Status
```
SYSTEM STATUS
==================================================
system_status: operational
last_update: 2024-01-15 14:30:25
recent_records_30_days: 45
total_records: 156
database_size_mb: 2.34
output_directory: outputs
database_path: road_accidents.db
```

## 🔍 Data Schema

The extracted data includes the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `date` | DateTime | Accident date |
| `title` | String | News article title |
| `text` | String | Article content (truncated) |
| `translated_text` | String | English translation |
| `district` | String | District name |
| `upazila` | String | Upazila name |
| `road` | String | Road/highway name |
| `specific_location` | String | Specific location details |
| `fatalities` | Integer | Number of fatalities |
| `injuries` | Integer | Number of injuries |
| `vehicle_types` | String | Comma-separated vehicle types |
| `severity` | String | Accident severity (fatal/major/minor) |
| `source_url` | String | Source article URL |
| `source_name` | String | News source name |
| `language` | String | Original language |
| `extraction_confidence` | Float | Extraction confidence score |

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Network errors**: Automatic retry with exponential backoff
- **Parsing errors**: Graceful degradation with fallback methods
- **Database errors**: Transaction rollback and logging
- **Translation errors**: Fallback to original text
- **Scheduler errors**: Automatic recovery and restart

## 📝 Logging

Detailed logs are written to `road_accident_analysis.log`:
- Execution time tracking
- Error details and stack traces
- Performance metrics
- Data processing statistics

## 🔒 Privacy and Ethics

- Only publicly available news articles are scraped
- No personal information is extracted or stored
- Respects robots.txt and rate limiting
- Data is used for research and analysis purposes only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- News sources for providing public access to accident reports
- Open-source NLP and ML libraries
- Research community for road safety analysis methodologies

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation and examples

---

**Note**: This system is designed for research and analysis purposes. Always verify extracted data and use responsibly for road safety initiatives.