#!/bin/bash
# Quick test runner for MonsoonBench

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "MonsoonBench Test Runner"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗${NC} pytest not found"
    echo "  Install with: pip install pytest pytest-mock"
    exit 1
fi

# Parse command line arguments
MODE=${1:-all}

case $MODE in
    unit)
        echo -e "${BLUE}Running unit tests...${NC}"
        pytest tests/unit/ -v
        ;;
    integration)
        echo -e "${BLUE}Running integration tests...${NC}"
        pytest tests/integration/ -v
        ;;
    quick)
        echo -e "${BLUE}Running quick tests (unit only)...${NC}"
        pytest tests/unit/ -v -x
        ;;
    coverage)
        echo -e "${BLUE}Running tests with coverage...${NC}"
        if command -v pytest-cov &> /dev/null; then
            pytest --cov=monsoonbench --cov-report=term-missing --cov-report=html
            echo ""
            echo -e "${GREEN}✓${NC} Coverage report generated in htmlcov/"
        else
            echo -e "${YELLOW}!${NC} pytest-cov not installed"
            echo "  Install with: pip install pytest-cov"
            exit 1
        fi
        ;;
    all)
        echo -e "${BLUE}Running all tests...${NC}"
        pytest -v
        ;;
    *)
        echo "Usage: $0 [all|unit|integration|quick|coverage]"
        echo ""
        echo "Options:"
        echo "  all          Run all tests (default)"
        echo "  unit         Run only unit tests"
        echo "  integration  Run only integration tests"
        echo "  quick        Run unit tests, stop on first failure"
        echo "  coverage     Run tests with coverage report"
        exit 1
        ;;
esac

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

exit $EXIT_CODE