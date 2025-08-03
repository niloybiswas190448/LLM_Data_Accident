#!/usr/bin/env python3
"""
Setup Script for Road Accident Analysis Pipeline
Helps users install and configure the system
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("="*60)
    print("🚗 Road Accident Analysis Pipeline Setup")
    print("="*60)
    print("Automated system for analyzing road accidents in Bangladesh")
    print("="*60)

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("\nDownloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
        print("You can download it manually later using:")
        print("python -c \"import nltk; nltk.download('punkt')\"")
        return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        "outputs",
        "logs",
        "data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def test_imports():
    """Test if all modules can be imported"""
    print("\nTesting imports...")
    
    modules = [
        "requests",
        "beautifulsoup4",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "transformers",
        "torch",
        "deep_translator",
        "schedule",
        "folium",
        "wordcloud"
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Warning: Some modules failed to import: {', '.join(failed_imports)}")
        print("You may need to install them manually:")
        for module in failed_imports:
            print(f"pip install {module}")
        return False
    
    return True

def test_pipeline_components():
    """Test pipeline components"""
    print("\nTesting pipeline components...")
    
    try:
        # Test basic imports
        from scrapers.news_scraper import NewsScraper
        from nlp.information_extractor import InformationExtractor
        from database.data_manager import DataManager
        from analysis.trend_analyzer import TrendAnalyzer
        from main import RoadAccidentPipeline
        
        print("✅ All pipeline components imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing pipeline components: {e}")
        return False

def create_config_file():
    """Create a sample configuration file if it doesn't exist"""
    print("\nChecking configuration...")
    
    if os.path.exists("config.py"):
        print("✅ Configuration file exists")
        return True
    
    print("⚠️ Configuration file not found")
    print("Please ensure config.py is present in the project directory")
    return False

def run_basic_test():
    """Run a basic test of the pipeline"""
    print("\nRunning basic test...")
    
    try:
        # Import and test pipeline
        from main import RoadAccidentPipeline
        
        pipeline = RoadAccidentPipeline()
        status = pipeline.get_system_status()
        
        print("✅ Pipeline test successful")
        print(f"System status: {status['system_status']}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        print("This might be due to missing internet connection or model downloads")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Run the pipeline:")
    print("   python main.py")
    
    print("\n2. Run with options:")
    print("   python main.py --help")
    
    print("\n3. Set up automated scheduling:")
    print("   python main.py --mode schedule")
    
    print("\n4. Generate a monthly report:")
    print("   python main.py --mode report --year 2024 --month 1")
    
    print("\n5. Check system status:")
    print("   python main.py --mode status")
    
    print("\n6. Run examples:")
    print("   python example_usage.py")
    
    print("\n7. Run tests:")
    print("   python test_pipeline.py")
    
    print("\n📚 Documentation:")
    print("- README.md: Complete documentation")
    print("- config.py: Configuration options")
    print("- example_usage.py: Usage examples")
    
    print("\n📁 Output files will be saved to:")
    print("- outputs/: Visualizations and reports")
    print("- road_accidents.db: Database")
    print("- road_accident_analysis.log: Logs")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed: Could not create directories")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n⚠️ Setup completed with warnings")
    
    # Test pipeline components
    if not test_pipeline_components():
        print("\n❌ Setup failed: Pipeline components not working")
        sys.exit(1)
    
    # Check configuration
    if not create_config_file():
        print("\n⚠️ Setup completed with warnings")
    
    # Run basic test
    run_basic_test()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()