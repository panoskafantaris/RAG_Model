#!/bin/bash
# Quick query script - easiest way to ask a question from terminal
# Usage: 
#   ./quick_query.sh "Your question here"
#   ./quick_query.sh "Πώς κάνω restart το PostgreSQL;"

BASE_URL="http://localhost:8000"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if message provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: No message provided${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 \"Your question here\""
    echo ""
    echo "Examples:"
    echo "  $0 \"Πώς κάνω restart το PostgreSQL;\""
    echo "  $0 \"How do I create a Linux user?\""
    exit 1
fi

MESSAGE="$1"

# Check if server is running
echo -e "${BLUE}Checking server...${NC}"
if ! curl -s -f "$BASE_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to server at $BASE_URL${NC}"
    echo "Make sure the server is running:"
    echo "  python setup_and_run.py --run"
    exit 1
fi

echo -e "${GREEN}✓ Server is running${NC}\n"

# Send query
echo -e "${CYAN}Question:${NC} $MESSAGE"
echo -e "${YELLOW}⏳ Waiting for response...${NC}\n"

# Make request and parse response
RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"role\":\"user\",\"content\":\"$MESSAGE\"}")

# Check if request was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to get response${NC}"
    exit 1
fi

# Extract answer using python
ANSWER=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('answer', 'No answer'))")

# Extract sources
SOURCES=$(echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
sources = data.get('sources', [])
if sources:
    print('\nSources:')
    for i, src in enumerate(sources, 1):
        print(f'  {i}. {src[\"source\"]} (relevance: {src[\"relevance_score\"]:.2f})')
")

# Print answer
echo -e "${GREEN}Answer:${NC}"
echo "$ANSWER"

# Print sources if available
if [ -n "$SOURCES" ]; then
    echo -e "${YELLOW}$SOURCES${NC}"
fi

echo ""