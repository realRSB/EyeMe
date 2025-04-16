#!/bin/bash

echo "🚀 Eyeme: Start Menu"
echo "---------------------"
echo "1. Run Main App (Real-time camera)"
echo "2. Run Streamlit Dashboard"
echo "3. Exit"
echo "---------------------"
read -p "Select an option (1-3): " choice

source venv/bin/activate

case $choice in
    1)
        echo "🟢 Running main camera app..."
        python3 -m app.main
        ;;
    2)
        echo "🌐 Launching Streamlit dashboard..."
        streamlit run app/dashboard.py
        ;;
    3)
        echo "👋 Exiting."
        exit 0
        ;;
    *)
        echo "❌ Invalid option. Please enter 1, 2, or 3."
        ;;
esac
