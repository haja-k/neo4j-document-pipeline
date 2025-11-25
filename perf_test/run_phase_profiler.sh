#!/bin/bash
# Quick start script for Neo4j Phase Profiler

echo "🔍 Neo4j Phase Profiler - Quick Start"
echo "====================================="
echo ""

# Check if we're in the right directory
if [ ! -f "neo4j_phase_profiler.py" ]; then
    echo "❌ Error: neo4j_phase_profiler.py not found"
    echo "Please run this script from the perf_test directory"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found"
    echo "Please install Python 3 to continue"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import requests, pandas, openpyxl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip install requests pandas openpyxl rich
fi

# Create output directory
mkdir -p test_results

# Run the profiler
echo ""
echo "🚀 Starting Neo4j Phase Profiler..."
echo ""
python3 neo4j_phase_profiler.py

echo ""
echo "✅ Profiling complete!"
echo "📊 Check test_results/ directory for Excel reports"
