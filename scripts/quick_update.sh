#!/bin/bash
# Simple interactive update script

echo "🔄 Git Remote Update Helper"
echo "=========================="

# Check git status
echo "📊 Current status:"
git status --short

echo ""
echo "Choose an option:"
echo "1) Commit current changes and update"
echo "2) Stash changes and update"  
echo "3) Just fetch (no local changes)"
echo "4) Cancel"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "💾 Committing changes..."
        git add .
        read -p "Enter commit message: " msg
        git commit -m "$msg"
        echo "📥 Pulling updates..."
        git pull --rebase origin main
        ;;
    2)
        echo "📦 Stashing changes..."
        git stash push -m "Auto-stash before update $(date)"
        echo "📥 Pulling updates..."
        git pull --rebase origin main
        echo "📤 Restoring stashed changes..."
        git stash pop
        ;;
    3)
        echo "📥 Fetching updates..."
        git fetch origin
        echo "📊 Comparison with remote:"
        git status
        echo "To merge: git pull --rebase origin main"
        ;;
    4)
        echo "❌ Cancelled"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo "✅ Update process complete!"
