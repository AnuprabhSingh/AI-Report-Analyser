#!/bin/bash
# Quick start script for Render + Vercel deployment

echo "🚀 Medical Interpreter - Render + Vercel Deployment"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Step 1: Committing deployment files...${NC}"
git add .
git status

echo ""
read -p "Commit these changes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "Configure for Render + Vercel deployment"
    echo -e "${GREEN}✓ Committed${NC}"
    
    echo ""
    echo -e "${BLUE}Step 2: Pushing to GitHub...${NC}"
    git push
    echo -e "${GREEN}✓ Pushed${NC}"
else
    echo "Skipping commit"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ Ready for deployment!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1️⃣  Deploy Backend to Render:"
echo "   → Go to: ${YELLOW}https://render.com${NC}"
echo "   → New Web Service → Connect GitHub"
echo "   → Select this repo"
echo "   → Environment: Docker"
echo "   → Dockerfile: ${YELLOW}Dockerfile.backend${NC}"
echo ""
echo "2️⃣  Deploy Frontend to Vercel:"
echo "   → Go to: ${YELLOW}https://vercel.com${NC}"
echo "   → Import Project from GitHub"
echo "   → Root Directory: ${YELLOW}frontend-react${NC}"
echo "   → Framework: Vite"
echo ""
echo "📚 Detailed Instructions:"
echo "   → See: ${YELLOW}DEPLOY_RENDER_VERCEL.md${NC}"
echo "   → Checklist: ${YELLOW}DEPLOYMENT_CHECKLIST.md${NC}"
echo ""
echo "🧪 Test locally first:"
echo "   → ${YELLOW}docker-compose up --build${NC}"
echo ""
