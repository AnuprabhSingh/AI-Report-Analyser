#!/bin/bash
# Quick deployment test script

echo "🧪 Testing deployment setup..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Docker
echo -n "Checking Docker... "
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Docker not installed${NC}"
    echo "Install from: https://www.docker.com/get-started"
fi

# Check Docker Compose
echo -n "Checking Docker Compose... "
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Docker Compose not installed${NC}"
fi

# Check Git
echo -n "Checking Git... "
if command -v git &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
    
    # Check if git repo initialized
    if [ -d .git ]; then
        echo -e "  ${GREEN}✓${NC} Git repository initialized"
    else
        echo -e "  ${YELLOW}! Git repository not initialized${NC}"
        echo "    Run: git init"
    fi
else
    echo -e "${RED}✗ Git not installed${NC}"
fi

# Check Node.js (for frontend)
echo -n "Checking Node.js... "
if command -v node &> /dev/null; then
    echo -e "${GREEN}✓${NC} $(node --version)"
else
    echo -e "${YELLOW}! Node.js not installed (needed for frontend development)${NC}"
fi

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} $(python3 --version)"
else
    echo -e "${RED}✗ Python not installed${NC}"
fi

echo ""
echo "📁 Checking project structure..."

# Check essential files
files=(
    "Dockerfile"
    "docker-compose.yml"
    "requirements.txt"
    "src/api.py"
    "models"
    ".env.example"
)

for file in "${files[@]}"; do
    if [ -e "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file missing"
    fi
done

echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Test locally with Docker:"
echo "   ${YELLOW}docker-compose up --build${NC}"
echo ""
echo "2. Push to GitHub:"
echo "   ${YELLOW}git init${NC}"
echo "   ${YELLOW}git add .${NC}"
echo "   ${YELLOW}git commit -m 'Initial commit'${NC}"
echo "   ${YELLOW}git branch -M main${NC}"
echo "   ${YELLOW}git remote add origin YOUR_REPO_URL${NC}"
echo "   ${YELLOW}git push -u origin main${NC}"
echo ""
echo "3. Deploy (choose one):"
echo "   • Render.com: Connect GitHub repo (easiest)"
echo "   • Railway.app: Connect GitHub repo"
echo "   • Vercel: For frontend only"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for detailed instructions"
