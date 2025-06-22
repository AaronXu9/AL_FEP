#!/bin/bash
# Force pull script with safety options

echo "🚨 FORCE PULL FROM REMOTE"
echo "========================"
echo "⚠️  WARNING: This will discard local changes!"
echo ""

# Show current status
echo "📊 Current status:"
git status --short | head -20
echo ""

echo "Choose force pull method:"
echo "1) Safe force pull (create backup branch first)"
echo "2) Nuclear option (discard everything, no backup)"
echo "3) Abort merge and force pull"
echo "4) Cancel"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "💾 Creating backup branch..."
        backup_name="backup-$(date +%Y%m%d-%H%M%S)"
        git branch "$backup_name"
        echo "✅ Backup created: $backup_name"
        
        echo "🔄 Fetching remote..."
        git fetch origin
        
        echo "🔥 Force pulling..."
        git reset --hard origin/main
        
        echo "✅ Force pull complete! Your backup is in branch: $backup_name"
        echo "   To restore backup: git checkout $backup_name"
        ;;
        
    2)
        echo "💀 Nuclear option selected..."
        read -p "Are you ABSOLUTELY sure? Type 'YES' to continue: " confirm
        if [ "$confirm" = "YES" ]; then
            echo "🔄 Fetching remote..."
            git fetch origin
            
            echo "💥 Resetting to remote..."
            git reset --hard origin/main
            
            echo "🧹 Cleaning untracked files..."
            git clean -fd
            
            echo "✅ Nuclear force pull complete!"
        else
            echo "❌ Cancelled"
            exit 0
        fi
        ;;
        
    3)
        echo "❌ Aborting current merge..."
        git merge --abort || echo "No merge to abort"
        
        echo "🔄 Fetching remote..."
        git fetch origin
        
        echo "🔥 Force pulling..."
        git reset --hard origin/main
        
        echo "✅ Force pull complete!"
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

echo ""
echo "📊 Final status:"
git status
echo ""
echo "🏁 Force pull operation completed!"
