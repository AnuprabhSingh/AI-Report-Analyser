#!/bin/bash
# Test script for CORS and API endpoints
# Run after deploying the CORS fix

BACKEND_URL="https://medical-interpreter-backend.onrender.com"
FRONTEND_URL="https://ai-report-analyser.vercel.app"

echo "🔍 Testing Medical Interpreter Backend"
echo "========================================"
echo ""

# Test 1: Health Check
echo "1️⃣  Testing /health endpoint..."
echo "   Command: curl -i ${BACKEND_URL}/health"
echo ""
curl -i "${BACKEND_URL}/health" \
  -H "Origin: ${FRONTEND_URL}" \
  -H "Access-Control-Request-Method: GET"
echo -e "\n"

# Test 2: Check CORS headers
echo "2️⃣  Checking CORS Headers..."
echo "   Looking for: Access-Control-Allow-Origin"
echo ""
curl -I "${BACKEND_URL}/health" \
  -H "Origin: ${FRONTEND_URL}"
echo -e "\n"

# Test 3: Test OPTIONS request (CORS preflight)
echo "3️⃣  Testing CORS Preflight Request..."
echo "   Command: curl -X OPTIONS with CORS headers"
echo ""
curl -X OPTIONS "${BACKEND_URL}/api/interpret" \
  -H "Origin: ${FRONTEND_URL}" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -v 2>&1 | grep -i "access-control"
echo -e "\n"

# Test 4: Get model comparison (if available)
echo "4️⃣  Testing /api/model-comparison endpoint..."
echo "   Command: curl -X GET ${BACKEND_URL}/api/model-comparison"
echo ""
curl -s "${BACKEND_URL}/api/model-comparison" \
  -H "Origin: ${FRONTEND_URL}" | head -c 500
echo -e "\n\n"

echo "✅ Testing Complete!"
echo ""
echo "📋 What to look for:"
echo "   ✓ Health check returns: {\"status\":\"healthy\",...}"
echo "   ✓ Response includes: Access-Control-Allow-Origin: ${FRONTEND_URL}"
echo "   ✓ Preflight returns 200 OK"
echo "   ✓ No CORS errors in responses"
echo ""
echo "❌ If you see CORS errors:"
echo "   1. Wait 2-3 minutes for Render to fully redeploy"
echo "   2. Check environment variable CORS_ORIGINS is set"
echo "   3. See CORS_DEPLOYMENT_FIX.md for detailed troubleshooting"
