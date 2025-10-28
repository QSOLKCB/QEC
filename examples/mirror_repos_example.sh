#!/usr/bin/env bash
# Example usage of the mirror_repos.sh script

echo "============================================"
echo "Repository Mirroring Script Example Usage"
echo "============================================"
echo ""

# Show script help/info
echo "Script location: $(readlink -f ./mirror_repos.sh)"
echo "Script permissions: $(ls -la ./mirror_repos.sh | cut -d' ' -f1)"
echo ""

echo "Environment variables (current settings):"
echo "  ORG_TEST=${ORG_TEST:-EmergentMonk}"
echo "  ORG_DEV=${ORG_DEV:-multimodalas}"
echo "  DEST_ORG=${DEST_ORG:-QSOLKCB}"
echo "  TMPDIR=${TMPDIR:-$HOME/repo_mirror}"
echo "  DRY_RUN=${DRY_RUN:-0}"
echo "  USE_HTTPS=${USE_HTTPS:-0}"
echo ""

echo "Example commands:"
echo ""
echo "1. Basic usage (requires GitHub CLI authentication):"
echo "   ./mirror_repos.sh"
echo ""
echo "2. Dry run mode (safe testing without pushing):"
echo "   DRY_RUN=1 ./mirror_repos.sh"
echo ""
echo "3. Custom organizations:"
echo "   ORG_TEST=my-test-org ORG_DEV=my-dev-org DEST_ORG=my-dest-org ./mirror_repos.sh"
echo ""
echo "4. CI/CD usage (HTTPS instead of SSH):"
echo "   USE_HTTPS=1 ./mirror_repos.sh"
echo ""
echo "5. Custom temporary directory:"
echo "   TMPDIR=/tmp/my-mirror DRY_RUN=1 ./mirror_repos.sh"
echo ""

echo "Dependencies check:"
if command -v gh >/dev/null 2>&1; then
    echo "  ✓ GitHub CLI (gh) is installed: $(gh --version | head -1)"
else
    echo "  ✗ GitHub CLI (gh) is NOT installed"
fi

if command -v git >/dev/null 2>&1; then
    echo "  ✓ Git is installed: $(git --version)"
else
    echo "  ✗ Git is NOT installed"
fi

echo ""
echo "To authenticate with GitHub CLI:"
echo "  gh auth login"
echo ""
echo "For more information, see docs/MIRROR_REPOS_GUIDE.md"
echo ""