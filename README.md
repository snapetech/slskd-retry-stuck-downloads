# slskd-retry-stuck-downloads

Utility to automatically retry and replace stuck Soulseek downloads managed by [slskd](https://github.com/slskd/slskd).

## Features

- **Auto-retry** - Automatically re-enqueues failed/stuck downloads
- **Alt-source search** - Finds alternative sources when original peer is offline
- **Auto-replace** - Automatically replaces stuck downloads with alternatives (within configurable size threshold)
- **Batch searching** - Parallel searches for 8x faster processing
- **Deduplication** - Removes duplicate downloads from queue

## Problem States Handled

- `Completed, TimedOut`
- `Completed, Errored`  
- `Completed, Rejected`
- `Completed, Aborted`

## Installation

```bash
pip install requests
```

## Configuration

Set these environment variables (or pass as CLI flags):

```bash
export SLSKD_API_KEY="your-api-key-here"
export SLSKD_BASE_URL="http://your-slskd-host:5030/api/v0"
```

You can find/create your API key in slskd's web UI under Settings → Options → Web.

## Usage

```bash
# Basic usage - retry stuck downloads (uses env vars)
python slskd_retry_stuck_downloads.py

# Or pass credentials directly
python slskd_retry_stuck_downloads.py --api-key YOUR_API_KEY --base-url http://localhost:5030/api/v0

# Auto-replace mode - automatically swap stuck downloads with alternatives
python slskd_retry_stuck_downloads.py --auto-replace

# Fast mode - shorter search timeout + parallel searches
python slskd_retry_stuck_downloads.py --auto-replace --search-timeout 5000 --batch-size 8
```

## Options

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--base-url` | `SLSKD_BASE_URL` | `http://localhost:5030/api/v0` | slskd API base URL |
| `--api-key` | `SLSKD_API_KEY` | (required) | slskd API key |
| `--auto` | - | `false` | Non-interactive mode (no prompts) |
| `--auto-replace` | - | `false` | Automatically replace stuck downloads |
| `--auto-replace-threshold` | - | `2.0` | Max size difference % for auto-replace |
| `--search-timeout` | - | `10000` | Search timeout in ms |
| `--batch-size` | - | `8` | Number of parallel searches |
| `--retry-before-replace` | - | `0` | Retry attempts before searching for alt |
| `--per-file-cooldown` | - | `30` | Seconds between retries per file |

## How It Works

1. Fetches all downloads from slskd API
2. Identifies "problem" downloads (stuck/failed states)
3. For each problem download:
   - Attempts to re-enqueue if peer is online
   - If peer offline, searches for alternative source
   - If alternative found within size threshold, auto-replaces
4. Batch processes searches in parallel for speed

## Development

This project was developed using [Cursor](https://cursor.sh/) with Claude (Opus 4) as the AI pair programming assistant. Code, documentation, and commit messages were collaboratively drafted through this process.

## License

MIT

