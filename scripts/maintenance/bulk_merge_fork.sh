#!/usr/bin/env bash
# Bulk merge & fork script
# Merges all repos from EmergentMonk → DEV
# and from multimodalas → TEST, then forks both into QSOLKCB.

set -euo pipefail

# Configuration
ORG1="EmergentMonk"
ORG2="multimodalas"
TARGET1="DEV"
TARGET2="TEST"
DEST="QSOLKCB"
REPO_LIMIT=200

# Create log file with timestamp
LOG_FILE="$HOME/bulk_merge_fork_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling function
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v gh &> /dev/null; then
        error_exit "GitHub CLI (gh) is not installed. Please install it first."
    fi
    
    if ! gh auth status &> /dev/null; then
        error_exit "GitHub CLI is not authenticated. Please run 'gh auth login' first."
    fi
    
    log "Prerequisites check passed."
}

# Create directory structure
create_directories() {
    log "Creating directory structure..."
    mkdir -p "$HOME/gh_repos/$TARGET1" "$HOME/gh_repos/$TARGET2"
    cd "$HOME/gh_repos"
    log "Working directory: $(pwd)"
}

# Clone repositories from an organization
clone_org_repos() {
    local org="$1"
    local target_dir="$2"
    
    log "Fetching repository list from $org..."
    
    # Get repository list and handle empty results
    local repos
    repos=$(gh repo list "$org" --limit "$REPO_LIMIT" --json name -q '.[].name' 2>/dev/null || true)
    
    if [[ -z "$repos" ]]; then
        log "WARNING: No repositories found or cannot access organization: $org"
        return 0
    fi
    
    local repo_count=0
    while IFS= read -r repo; do
        if [[ -n "$repo" ]]; then
            log "→ Cloning $org/$repo to $target_dir/$repo"
            if gh repo clone "$org/$repo" "$target_dir/$repo" 2>>"$LOG_FILE"; then
                ((repo_count++))
                log "   ✓ Successfully cloned $repo"
            else
                log "   ✗ Failed to clone $repo"
            fi
        fi
    done <<< "$repos"
    
    log "Cloned $repo_count repositories from $org to $target_dir"
}

# Fork repositories to destination organization
fork_repos() {
    log "Starting fork process to $DEST organization..."
    
    local fork_count=0
    local failed_forks=0
    
    # Process all repositories in both target directories
    for target_dir in "$TARGET1" "$TARGET2"; do
        if [[ ! -d "$target_dir" ]]; then
            log "WARNING: Directory $target_dir does not exist, skipping..."
            continue
        fi
        
        for path in "$target_dir"/*; do
            if [[ -d "$path" ]]; then
                local repo
                repo=$(basename "$path")
                log "→ Processing fork for $repo under $DEST..."
                
                if cd "$path" 2>/dev/null; then
                    # Try to create fork
                    if gh repo fork "$DEST/$repo" --remote=true --default-branch-only 2>>"$LOG_FILE"; then
                        ((fork_count++))
                        log "   ✓ Successfully forked $repo to $DEST"
                    else
                        ((failed_forks++))
                        log "   ✗ Failed to fork $repo to $DEST"
                    fi
                    cd - >/dev/null
                else
                    log "   ✗ Cannot access directory: $path"
                    ((failed_forks++))
                fi
            fi
        done
    done
    
    log "Fork process completed: $fork_count successful, $failed_forks failed"
}

# Main execution
main() {
    log "Starting bulk merge & fork script"
    log "Configuration: $ORG1 → $TARGET1, $ORG2 → $TARGET2, forks → $DEST"
    
    check_prerequisites
    create_directories
    
    # Clone repositories from both organizations
    clone_org_repos "$ORG1" "$TARGET1"
    clone_org_repos "$ORG2" "$TARGET2"
    
    # Fork all repositories to destination
    fork_repos
    
    log "✅ Bulk merge & fork script completed successfully"
    log "Log file saved to: $LOG_FILE"
}

# Run main function
main "$@"