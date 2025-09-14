#!/bin/bash

echo "🔍 Testing Mentra Connection Status"
echo "===================================="
echo ""

# Check local server
echo "1. Checking local server..."
curl -s http://localhost:4000/api/stream/status | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'   Connected: {data.get(\"connected\", False)}'); print(f'   Status: {data.get(\"status\", \"unknown\")}'); print(f'   Message: {data.get(\"message\", \"\")}');"

echo ""
echo "2. Checking ngrok tunnel..."
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; data = json.load(sys.stdin); print(data['tunnels'][0]['public_url'] if data.get('tunnels') else 'No tunnel found')")
echo "   Public URL: $NGROK_URL"

echo ""
echo "3. Next steps:"
echo "   ✓ Ensure glasses are powered on and connected to MentraHackathon WiFi"
echo "   ✓ Open MentraOS app and navigate to your app: com.table46.real"
echo "   ✓ Update webhook URL in console.mentra.glass to: $NGROK_URL/webhook"
echo ""
echo "Press Ctrl+C to exit monitoring..."
echo ""
echo "Monitoring for connections (will update when glasses connect)..."

# Monitor in a loop
while true; do
    STATUS=$(curl -s http://localhost:4000/api/stream/status | python3 -c "import sys, json; data = json.load(sys.stdin); print('✅ CONNECTED' if data.get('connected') else '⏳ Waiting...')")
    echo -ne "\r$STATUS"
    
    if [[ "$STATUS" == *"CONNECTED"* ]]; then
        echo ""
        echo ""
        echo "🎉 Glasses connected successfully!"
        echo "You can now start streaming from http://localhost:3000"
        break
    fi
    
    sleep 2
done
