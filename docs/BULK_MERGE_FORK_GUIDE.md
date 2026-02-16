# Bulk Merge & Fork Script

## Overview

The `bulk_merge_fork.sh` script automates the process of cloning repositories from multiple GitHub organizations and creating forks in a destination organization.

## Features

- **Multi-Organization Support**: Clones repos from EmergentMonk and multimodalas organizations
- **Organized Directory Structure**: Separates cloned repos into DEV and TEST directories
- **Automated Forking**: Creates forks of all cloned repositories in the QSOLKCB organization
- **Comprehensive Logging**: Detailed logs with timestamps saved to file
- **Error Handling**: Robust error checking and graceful failure handling
- **Prerequisites Validation**: Checks for GitHub CLI installation and authentication

## Prerequisites

1. **GitHub CLI (gh)**: Must be installed and available in PATH
   ```bash
   # Install GitHub CLI (example for Ubuntu/Debian)
   curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
   sudo apt update
   sudo apt install gh
   ```

2. **GitHub Authentication**: Must be authenticated with appropriate permissions
   ```bash
   gh auth login
   ```

3. **Organization Access**: Must have access to read from source organizations (EmergentMonk, multimodalas) and create repos in destination organization (QSOLKCB)

## Usage

### Basic Usage

```bash
./bulk_merge_fork.sh
```

### Directory Structure Created

```
$HOME/gh_repos/
├── DEV/           # Repositories from EmergentMonk
│   ├── repo1/
│   ├── repo2/
│   └── ...
└── TEST/          # Repositories from multimodalas  
    ├── repo1/
    ├── repo2/
    └── ...
```

### Configuration

The script uses the following default configuration (modify the script to change):

- **Source Organizations**: 
  - `EmergentMonk` → cloned to `DEV/` directory
  - `multimodalas` → cloned to `TEST/` directory
- **Destination Organization**: `QSOLKCB`
- **Repository Limit**: 200 per organization
- **Working Directory**: `$HOME/gh_repos`

## Script Workflow

1. **Prerequisites Check**: Validates GitHub CLI installation and authentication
2. **Directory Setup**: Creates necessary directory structure
3. **Repository Cloning**: 
   - Fetches repository lists from source organizations
   - Clones each repository to appropriate target directory
4. **Fork Creation**: Creates forks of all cloned repositories in destination organization
5. **Logging**: Generates comprehensive log file with operation results

## Logging

The script creates detailed log files with timestamps:
- **Log Location**: `$HOME/bulk_merge_fork_YYYYMMDD_HHMMSS.log`
- **Log Content**: 
  - Operation timestamps
  - Success/failure status for each repository
  - Error messages and warnings
  - Summary statistics

## Error Handling

The script includes robust error handling:
- **Prerequisite Validation**: Checks for required tools and authentication
- **Network Errors**: Handles GitHub API failures gracefully
- **Permission Issues**: Reports repositories that cannot be accessed
- **Directory Problems**: Validates directory creation and navigation
- **Fork Conflicts**: Handles existing forks and naming conflicts

## Example Output

```
[2024-10-28 16:45:30] Starting bulk merge & fork script
[2024-10-28 16:45:30] Configuration: EmergentMonk → DEV, multimodalas → TEST, forks → QSOLKCB
[2024-10-28 16:45:30] Checking prerequisites...
[2024-10-28 16:45:31] Prerequisites check passed.
[2024-10-28 16:45:31] Creating directory structure...
[2024-10-28 16:45:31] Working directory: /home/user/gh_repos
[2024-10-28 16:45:31] Fetching repository list from EmergentMonk...
[2024-10-28 16:45:32] → Cloning EmergentMonk/repo1 to DEV/repo1
[2024-10-28 16:45:35] ✓ Successfully cloned repo1
...
[2024-10-28 16:50:15] ✅ Bulk merge & fork script completed successfully
[2024-10-28 16:50:15] Log file saved to: /home/user/bulk_merge_fork_20241028_164530.log
```

## Troubleshooting

### Common Issues

1. **GitHub CLI Not Found**
   - Ensure GitHub CLI is installed and in PATH
   - Verify installation: `gh --version`

2. **Authentication Failed**
   - Re-authenticate: `gh auth login`
   - Check token permissions: `gh auth status`

3. **Organization Access Denied**
   - Verify you have read access to source organizations
   - Check if organizations exist and are accessible

4. **Fork Creation Failed**
   - Ensure you have permission to create repositories in destination organization
   - Check if repository names conflict with existing repos

### Getting Help

For issues with the script:
1. Check the generated log file for detailed error information
2. Verify GitHub CLI configuration and permissions
3. Test manual operations with `gh` commands to isolate issues

## Security Considerations

- **Token Permissions**: Ensure GitHub token has minimal required permissions
- **Organization Access**: Only grant access to necessary organizations
- **Local Storage**: Be aware that repositories are cloned locally (may contain sensitive data)
- **Fork Visibility**: Consider the visibility settings of created forks

## License

This script is part of the fusion-qec-sim project and follows the same licensing terms.