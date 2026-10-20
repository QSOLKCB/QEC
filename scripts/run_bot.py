"""
Main entry point for the integrated QEC IRC bot.

This script can be run directly to start the bot.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from integrated_bot import main

if __name__ == '__main__':
    sys.exit(main())
