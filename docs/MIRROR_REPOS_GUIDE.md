# Repository Mirroring Script

This script (`mirror_repos.sh`) provides comprehensive repository mirroring and synchronization functionality between multiple GitHub organizations.

## Features

- **Multi-Organization Sync**: Sync repositories from test and development organizations to a destination organization
- **Branch Management**: Automatically creates and manages TEST, DEV, and BUILD branches
- **Efficient Cloning**: Uses `--filter=blob:none` for faster cloning of large repositories
- **Fork Management**: Automatically creates forks in the destination organization
- **Merge Strategies**: Intelligent merging of TEST and DEV branches into BUILD branch
- **Flexible Configuration**: Supports both HTTPS and SSH authentication
- **Dry Run Mode**: Test the script without actually pushing changes

## Configuration

The script uses environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `ORG_TEST` | `EmergentMonk` | Source organization for test repositories |
| `ORG_DEV` | `multimodalas` | Source organization for development repositories |
| `DEST_ORG` | `QSOLKCB` | Destination organization for forks |
| `TMPDIR` | `$HOME/repo_mirror` | Directory for local repository clones |
| `DRY_RUN` | `0` | Set to `1` to test without pushing (dry run mode) |
| `USE_HTTPS` | `0` | Set to `1` to use HTTPS instead of SSH for git operations |

## Dependencies

- `gh` (GitHub CLI) - for repository operations
- `git` - for version control operations

## Usage

### Basic Usage

```bash
# Run with default settings
./mirror_repos.sh
```

### Dry Run Mode

```bash
# Test the script without making changes
DRY_RUN=1 ./mirror_repos.sh
```

### Custom Organizations

```bash
# Use custom organizations
ORG_TEST=my-test-org ORG_DEV=my-dev-org DEST_ORG=my-dest-org ./mirror_repos.sh
```

### CI/CD Usage

```bash
# Use HTTPS for CI environments
USE_HTTPS=1 ./mirror_repos.sh
```

## How It Works

1. **Sync TEST Branch**: Clones all repositories from the test organization and creates/updates TEST branches
2. **Sync DEV Branch**: Clones all repositories from the development organization and creates/updates DEV branches
3. **Create Forks**: Ensures forks exist in the destination organization
4. **Build Branch Creation**: Creates BUILD branches by merging TEST and DEV branches
5. **Push Changes**: Pushes all branches to the destination organization

## Branch Strategy

- **TEST**: Contains the latest code from the test organization
- **DEV**: Contains the latest code from the development organization  
- **BUILD**: Merged branch containing changes from both TEST and DEV branches

## Error Handling

The script includes comprehensive error handling:
- Checks for required dependencies
- Handles missing repositories gracefully
- Continues processing even if individual operations fail
- Provides detailed logging with timestamps

## Logging

All operations are logged with timestamps for tracking and debugging:

```
[2025-10-28 18:50:31] == SYNC TEST from EmergentMonk ==
[2025-10-28 18:50:32] Processing EmergentMonk/repo1 → TEST
[2025-10-28 18:50:35] == SYNC DEV from multimodalas ==
[2025-10-28 18:50:36] Processing multimodalas/repo1 → DEV
[2025-10-28 18:50:40] == BUILD ==
[2025-10-28 18:50:41] [BUILD] repo1 updated
[2025-10-28 18:50:42] ✅ Done.
```

## Security Considerations

- The script requires GitHub CLI authentication
- SSH keys or GitHub tokens must be properly configured
- Use `DRY_RUN=1` to test before making actual changes
- Ensure proper permissions on the destination organization