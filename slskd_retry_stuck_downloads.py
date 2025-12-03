#!/usr/bin/env python3
"""
slskd_retry_stuck_downloads.py

Utility to retry "problem" Soulseek downloads managed by slskd, and optionally
suggest alternative sources for tracks that keep failing.

Problem downloads are those in "Completed, TimedOut", "Completed, Errored",
"Completed, Rejected", or "Completed, Aborted" states.

The script:
- Takes an initial snapshot of problem downloads.
- Walks that queue, re-enqueuing items one-by-one.
- Enforces a per-file cooldown between retries.
- Counts retry cycles per track and, after a threshold, tries to find a better
  source using the slskd search API, prompting you when it finds one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Iterable

import requests


# ---------------------- Defaults / Tunables ---------------------- #

DEFAULT_BASE_URL = os.environ.get("SLSKD_BASE_URL", "http://localhost:5030/api/v0")
DEFAULT_REQUEST_TIMEOUT = 10.0
DEFAULT_MAX_HTTP_RETRIES = 4
DEFAULT_BACKOFF_FACTOR = 1.5

# Minimum seconds between retry attempts for the same track
DEFAULT_PER_FILE_COOLDOWN = 30.0

# How often (seconds) to refresh the snapshot from slskd and drop
# items that no longer need retry.
DEFAULT_SNAPSHOT_REFRESH_INTERVAL = 60.0

# How many retry *cycles* for a given track before we try to find
# an alternative source. 1 == first failure triggers alt search.
DEFAULT_RETRY_BEFORE_REPLACE = 0

# Max relative size difference for an alt source (e.g. 0.50 == 50%)
DEFAULT_MAX_SIZE_DIFF = 0.50

# Transfer states considered "problematic" and subject to retry
PROBLEM_STATES = {
    "Completed, TimedOut",
    "Completed, Errored",
    "Completed, Rejected",
    "Completed, Aborted",
    "Completed, Cancelled",  # User-cancelled items (often due to being stuck)
}


# ---------------------- Data structures ---------------------- #

@dataclass(frozen=True)
class DownloadKey:
    username: str
    directory: str
    filename: str

    def track_key(self) -> Tuple[str, str]:
        """Key representing 'this track' regardless of source user."""
        return (self.directory.lower(), self.filename.lower())


@dataclass(frozen=True)
class DownloadItem:
    key: DownloadKey
    id: str
    size: int
    state: str
    ext: str

    def display_name(self) -> str:
        return f"{self.key.username} :: {self.key.directory}\\{self.key.filename}"


# ---------------------- API helpers ---------------------- #

class SlskdSession:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # slskd uses X-Api-Key header
        self.session.headers.update({"X-Api-Key": api_key})
        self.timeout = timeout

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def get(self, path: str, **kwargs) -> requests.Response:
        return self.session.get(self._url(path), timeout=self.timeout, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        return self.session.post(self._url(path), timeout=self.timeout, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        return self.session.delete(self._url(path), timeout=self.timeout, **kwargs)


def fetch_all_downloads(slskd: SlskdSession) -> List[dict]:
    """
    GET /transfers/downloads -> list of Transfer objects.
    """
    resp = slskd.get("/transfers/downloads")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected downloads payload type: {type(data).__name__}")
    return data


def iter_download_items(raw_downloads: List[dict]) -> Iterable[DownloadItem]:
    """
    Flatten the slskd downloads payload into DownloadItem instances.
    """
    for transfer in raw_downloads:
        username = transfer.get("username") or ""
        directories = transfer.get("directories") or []
        for d in directories:
            dir_path = d.get("directory") or ""
            files = d.get("files") or []
            for f in files:
                state = f.get("state") or ""
                file_id = f.get("id")
                filename = f.get("filename") or ""
                size = int(f.get("size") or 0)
                ext = (f.get("extension") or "")
                if not ext and "." in filename:
                    ext = filename.rsplit(".", 1)[-1]
                ext = ext.lower()
                if not file_id:
                    # Should not happen, but skip if it does
                    continue
                key = DownloadKey(username=username, directory=dir_path, filename=filename)
                yield DownloadItem(
                    key=key,
                    id=file_id,
                    size=size,
                    state=state,
                    ext=ext,
                )


def collect_states(slskd: SlskdSession) -> List[str]:
    raw = fetch_all_downloads(slskd)
    states = sorted({item.state for item in iter_download_items(raw)})
    return states


def build_problem_queue(raw_downloads: List[dict], debug: bool = False) -> List[DownloadItem]:
    """
    From all downloads, build an ordered list of "problem" items to process.
    Deduplicates by DownloadKey.
    """
    seen: set[DownloadKey] = set()
    queue: List[DownloadItem] = []
    state_counts: dict[str, int] = {}
    skipped_examples: dict[str, str] = {}  # state -> example filename
    
    for item in iter_download_items(raw_downloads):
        state_counts[item.state] = state_counts.get(item.state, 0) + 1
        
        if item.state not in PROBLEM_STATES:
            if item.state not in skipped_examples:
                skipped_examples[item.state] = item.key.filename
            continue
        if item.key in seen:
            continue
        seen.add(item.key)
        queue.append(item)
    
    if debug:
        print(f"[DEBUG] Download states breakdown:")
        for state, count in sorted(state_counts.items()):
            in_problem = "✓" if state in PROBLEM_STATES else "✗"
            example = f" (e.g. {os.path.basename(skipped_examples.get(state, ''))})" if state not in PROBLEM_STATES else ""
            print(f"[DEBUG]   {in_problem} {state}: {count}{example}")
    
    return queue


def reenqueue_download(slskd: SlskdSession, item: DownloadItem) -> Tuple[bool, Optional[str]]:
    """
    POST /transfers/downloads/{username} with the original id/filename/directory.

    Returns (success, reason):
      success=True  -> re-enqueued
      success=False -> reason is "offline" or error message
    """
    url_path = f"/transfers/downloads/{item.key.username}"
    payload = [{
        "id": item.id,
        "filename": item.key.filename,
        "size": item.size,
        "directory": item.key.directory,
    }]
    delay = 1.0
    last_error: Optional[str] = None

    for attempt in range(DEFAULT_MAX_HTTP_RETRIES):
        try:
            resp = slskd.post(url_path, json=payload)
        except requests.RequestException as exc:
            last_error = f"request error: {exc}"
            if attempt == DEFAULT_MAX_HTTP_RETRIES - 1:
                return False, last_error
            time.sleep(delay)
            delay *= DEFAULT_BACKOFF_FACTOR
            continue

        text = (resp.text or "").strip()

        # Treat 200/201/202/204 as success
        if resp.status_code in (200, 201, 202, 204):
            return True, None

        # Special-case: offline peer message embedded in 500
        if "appears to be offline" in text.lower():
            return False, "offline"

        # Backoff on 429 and transient 5xx
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < DEFAULT_MAX_HTTP_RETRIES - 1:
            last_error = f"{resp.status_code}: {text or resp.reason}"
            time.sleep(delay)
            delay *= DEFAULT_BACKOFF_FACTOR
            continue

        # Other non-success: don't retry further
        last_error = f"{resp.status_code}: {text or resp.reason}"
        break

    return False, last_error or "unknown error"


# ---------------------- Search / alt-source helpers ---------------------- #

def clean_track_title(filename: str, strip_artist: str = None) -> str:
    """
    Best-effort cleanup of a track filename into a search-friendly title.
    Strips track numbers, quality info, special characters, and optionally artist prefix.
    """
    name = filename
    
    # Extract just the base filename if path separators are present
    if '\\' in name or '/' in name:
        name = re.split(r'[\\/]', name)[-1]
    
    # Strip extension
    if "." in name:
        name = name.rsplit(".", 1)[0]

    # Replace underscores with spaces
    name = name.replace("_", " ")

    # Strip quality/bitrate info: "(FLAC 1077 kbps)", "[FLAC]", "(320kbps)", etc.
    name = re.sub(r"\s*\(?\[?(?:FLAC|MP3|AAC|ALAC|WAV|OGG|WMA)[\s\d]*(?:kbps|kHz|bit)?\]?\)?", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*\(?\[?\d+\s*kbps\]?\)?", "", name, flags=re.IGNORECASE)
    
    # Strip year patterns that might interfere: "(2011)", "[1997]"
    # But be careful not to strip if it's the only content
    if len(re.sub(r"\s*[\(\[]?\d{4}[\)\]]?\s*", "", name).strip()) > 3:
        name = re.sub(r"\s*[\(\[]?\d{4}[\)\]]?\s*", " ", name)

    # Drop leading disc-track numbers like "1-17 ", "2-03 ", "CD-01 -", then standalone "01 - ", "0106 - " etc.
    name = re.sub(r"^(?:CD-?)?\d{1,2}-[0-9]{1,4}[\s.\-)_]+", "", name, flags=re.IGNORECASE).strip()
    # Then handle remaining standalone track numbers: "01 - Song" or "01 Song" or "01. Song"
    name = re.sub(r"^[0-9]{1,4}[\s.\-)_]+", "", name).strip()
    
    # Strip track numbers in middle of string: "Artist - Album - 01 - Song" -> "Artist - Album - Song"
    # This handles "- 01 -", "- 07 -", etc.
    name = re.sub(r"\s+-\s+\d{1,3}\s+-\s+", " - ", name)
    
    # Strip "Artist - " prefix if we already have the artist from directory
    if strip_artist:
        # Pattern: "Artist - ", "Artist feat. X - ", "Artist with X - ", "Artist & X - "
        artist_pattern = re.escape(strip_artist) + r"(?:\s+(?:feat\.?|featuring|with|&|and|vs\.?)[^-]*)?\s*-\s*"
        name = re.sub(f"^{artist_pattern}", "", name, flags=re.IGNORECASE).strip()
        # Also strip if artist appears after album: "Album - Artist - Song" (less common)
        name = re.sub(rf"\s*-\s*{re.escape(strip_artist)}\s*-\s*", " - ", name, flags=re.IGNORECASE)

    # Strip common junk suffixes
    name = re.sub(r"\s*[\(\[]?(?:official|music|video|audio|lyric|lyrics|explicit|clean|remaster|remastered|remix|edit|version|ver\.?)[\)\]]?\s*$", "", name, flags=re.IGNORECASE)
    
    # Strip parenthetical content that's just noise (but keep meaningful stuff like "(feat. X)")
    # Only strip if it's not "feat" related
    name = re.sub(r"\s*\([^)]*(?:remix|edit|mix|version|ver\.?|remaster|instrumental|acoustic|live|demo)\)", "", name, flags=re.IGNORECASE)

    # Collapse whitespace and extra dashes
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\s*-\s*-\s*", " - ", name)  # Fix double dashes from removals
    name = re.sub(r"^\s*-\s*|\s*-\s*$", "", name)  # Strip leading/trailing dashes

    return name.strip()


def infer_artist_from_directory(directory: str) -> Optional[str]:
    """
    Try to infer artist from the remote directory path.
    E.g. "flac\\Lil Jon & The East Side Boyz\\2004 - Crunk Juice"
        -> artist "Lil Jon & The East Side Boyz"
    """
    parts = [p for p in re.split(r"[\\/]", directory) if p]
    if len(parts) >= 2:
        return parts[-2]
    return None


def build_search_text(item: DownloadItem) -> str:
    """
    Heuristic search text: "artist track" (no extension - we filter results by ext).
    """
    artist = infer_artist_from_directory(item.key.directory)
    # Pass artist to clean_track_title so it can strip "Artist - " prefix
    title = clean_track_title(item.key.filename, strip_artist=artist)
    pieces = []
    if artist:
        pieces.append(artist)
    if title:
        pieces.append(title)
    # Don't include extension in search - it can cause zero results
    # We filter by extension when evaluating candidates instead
    return " ".join(pieces).strip()


DEFAULT_SEARCH_TIMEOUT_MS = 10000  # 10 seconds - faster than default 15s
DEFAULT_MAX_CONCURRENT_SEARCHES = 2  # slskd only processes ~2 concurrent searches


def start_search(slskd: SlskdSession, search_text: str, search_timeout_ms: int = DEFAULT_SEARCH_TIMEOUT_MS) -> Optional[str]:
    """Start a search and return the search_id (non-blocking)."""
    try:
        resp = slskd.post("/searches", json={
            "searchText": search_text,
            "searchTimeout": search_timeout_ms,
        })
        if resp.status_code >= 400:
            return None
        return resp.json().get("id")
    except requests.RequestException:
        return None


def poll_search(slskd: SlskdSession, search_id: str, force_responses: bool = False, debug: bool = False) -> Tuple[bool, Optional[List]]:
    """Poll a search. Returns (is_complete, responses_or_None).
    If force_responses=True, wait up to 10s extra for completion if we have responseCount."""
    try:
        # Fetch with includeResponses=true to get responses inline
        resp = slskd.get(f"/searches/{search_id}", params={"includeResponses": "true"})
        if resp.status_code >= 400:
            return False, None
        data = resp.json()
        state = data.get("state", "")
        is_complete = data.get("isComplete", False) or "Completed" in state
        response_count = data.get("responseCount", 0)
        responses = data.get("responses", [])
        
        if debug:
            print(f"[DEBUG] Search {search_id}: state={state}, isComplete={data.get('isComplete')}, responseCount={response_count}, inline_responses={len(responses)}")
        
        # If complete, we're done
        if is_complete:
            return True, responses
        
        # If force_responses requested (timeout case) and we have responses inline, return them
        if force_responses and responses:
            return True, responses
        
        # If force_responses and we have responseCount but no inline responses,
        # wait a bit longer for the search to complete (inline responses only work for complete searches)
        if force_responses and response_count > 0 and not responses:
            if debug:
                print(f"[DEBUG] Search {search_id}: has {response_count} responses but not complete, waiting...")
            # Wait up to 10 more seconds for completion
            for _ in range(10):
                time.sleep(1)
                resp2 = slskd.get(f"/searches/{search_id}", params={"includeResponses": "true"})
                if resp2.status_code >= 400:
                    continue
                data2 = resp2.json()
                state2 = data2.get("state", "")
                is_complete2 = data2.get("isComplete", False) or "Completed" in state2
                responses2 = data2.get("responses", [])
                if is_complete2 and responses2:
                    if debug:
                        print(f"[DEBUG] Search {search_id}: completed after extra wait, {len(responses2)} responses")
                    return True, responses2
        
        return False, None
    except requests.RequestException as e:
        if debug:
            print(f"[DEBUG] Search {search_id}: poll error: {e}")
        return False, None


def sliding_window_search(
    slskd: SlskdSession,
    search_items: List[Tuple[DownloadItem, str]],  # (item, search_text)
    max_concurrent: int = DEFAULT_MAX_CONCURRENT_SEARCHES,
    search_timeout_ms: int = DEFAULT_SEARCH_TIMEOUT_MS,
    per_search_timeout: float = 15.0,
) -> Dict[DownloadItem, Optional[List]]:
    """
    Perform searches with sliding window - keep max_concurrent searches running at all times.
    As soon as one finishes, start the next. Much more efficient than fixed batches.
    Returns dict mapping item -> responses (or None/[] if failed).
    """
    results: Dict[DownloadItem, Optional[List]] = {}
    
    if not search_items:
        return results
    
    print(f"[SEARCH] Processing {len(search_items)} searches (max {max_concurrent} concurrent)...")
    
    pending = list(search_items)  # Queue of items waiting to be searched
    active: Dict[str, Tuple[DownloadItem, float]] = {}  # search_id -> (item, start_time)
    poll_interval = 1.0
    
    while pending or active:
        # Start new searches up to max_concurrent
        while pending and len(active) < max_concurrent:
            item, search_text = pending.pop(0)
            search_id = start_search(slskd, search_text, search_timeout_ms)
            if search_id:
                active[search_id] = (item, time.time())
                print(f"[SEARCH] Started: '{clean_track_title(item.key.filename)}'")
            else:
                results[item] = []
                print(f"[SEARCH] Failed to start: '{clean_track_title(item.key.filename)}'")
        
        if not active:
            break
        
        time.sleep(poll_interval)
        
        # Check active searches
        completed = []
        now = time.time()
        for search_id, (item, start_time) in active.items():
            elapsed = now - start_time
            timed_out = elapsed >= per_search_timeout
            
            # Try to get responses (force if timed out, debug on timeout to see what's happening)
            is_complete, responses = poll_search(slskd, search_id, force_responses=timed_out, debug=timed_out)
            
            if is_complete or timed_out:
                results[item] = responses if responses else []
                completed.append(search_id)
                status = "✓" if responses else "✗"
                count = len(responses) if responses else 0
                print(f"[SEARCH] {status} '{clean_track_title(item.key.filename)}': {count} responses ({elapsed:.0f}s)")
        
        for search_id in completed:
            del active[search_id]
    
    found = sum(1 for r in results.values() if r)
    print(f"[SEARCH] Done: {found}/{len(results)} searches found results")
    return results


def perform_search(slskd: SlskdSession, search_text: str, max_wait: float = 30.0, search_timeout_ms: int = DEFAULT_SEARCH_TIMEOUT_MS) -> Optional[object]:
    """
    POST /searches { "searchText": search_text }
    Then poll GET /searches/{id} until complete or timeout.
    Returns either:
      - a list of response objects, or
      - a dict that may contain a 'responses' or 'items' array.
    """
    # Start search with configurable timeout
    try:
        resp = slskd.post("/searches", json={
            "searchText": search_text,
            "searchTimeout": search_timeout_ms,
        })
    except requests.RequestException as exc:
        print(f"[ALT] search_text request failed for '{search_text}': {exc}")
        return None

    if resp.status_code >= 400:
        print(f"[ALT] search_text failed: {resp.status_code} {resp.text.strip()}")
        return None

    state = resp.json()
    search_id = state.get("id")
    if not search_id:
        print(f"[ALT] search_text returned no id for '{search_text}'")
        return None

    # Debug: show initial search state
    print(f"[ALT-DEBUG] Initial search state keys: {list(state.keys())}")
    if "state" in state:
        print(f"[ALT-DEBUG] search state={state.get('state')}")
    if "responseCount" in state:
        print(f"[ALT-DEBUG] initial responseCount={state.get('responseCount')}")
    if "fileCount" in state:
        print(f"[ALT-DEBUG] initial fileCount={state.get('fileCount')}")

    # Poll until search is complete or we have responses, or timeout
    poll_interval = 2.0
    elapsed = 0.0
    result = []
    
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        
        # Check search state - include responses in the request
        try:
            state_resp = slskd.get(f"/searches/{search_id}", params={"includeResponses": "true"})
            if state_resp.status_code >= 400:
                print(f"[ALT-DEBUG] GET /searches/{search_id} failed: {state_resp.status_code}")
                continue
            state_data = state_resp.json()
            response_count = state_data.get("responseCount", 0)
            file_count = state_data.get("fileCount", 0)
            search_state = state_data.get("state", "")
            is_complete = state_data.get("isComplete", False)
            
            print(f"[ALT-DEBUG] poll {elapsed:.0f}s: state={search_state}, isComplete={is_complete}, "
                  f"responseCount={response_count}, fileCount={file_count}")
            
            # Responses only appear AFTER search is complete!
            # If search is complete, fetch responses
            if is_complete or "Completed" in search_state:
                responses_data = state_data.get("responses", [])
                print(f"[ALT-DEBUG] Search complete! responses_len={len(responses_data) if hasattr(responses_data, '__len__') else 'N/A'}")
                if responses_data and len(responses_data) > 0:
                    result = responses_data
                    print(f"[ALT-DEBUG] Got {len(result)} responses from search state")
                break
            
            # Not complete yet - keep waiting
            if search_state in ("Cancelled",):
                break
                
        except requests.RequestException as exc:
            print(f"[ALT-DEBUG] poll error: {exc}")
    
    # Final fetch if we haven't already - must wait for completion
    if not result:
        print(f"[ALT-DEBUG] Polling timed out without completion, doing final fetch...")
        try:
            final_state = slskd.get(f"/searches/{search_id}", params={"includeResponses": "true"})
            if final_state.status_code < 400:
                final_data = final_state.json()
                is_complete_final = final_data.get("isComplete", False)
                responses_from_state = final_data.get("responses", [])
                print(f"[ALT-DEBUG] Final: isComplete={is_complete_final}, responses_len={len(responses_from_state) if hasattr(responses_from_state, '__len__') else 'N/A'}")
                if responses_from_state and len(responses_from_state) > 0:
                    result = responses_from_state
                    print(f"[ALT-DEBUG] Final fetch got {len(result)} responses")
        except requests.RequestException as e:
            print(f"[ALT-DEBUG] Final fetch error: {e}")
    
    # Debug: show raw response structure
    if isinstance(result, dict):
        print(f"[ALT-DEBUG] Final API response is dict with keys: {list(result.keys())}")
    elif isinstance(result, list):
        print(f"[ALT-DEBUG] Final API response is list with {len(result)} items")
    else:
        print(f"[ALT-DEBUG] Final API response type: {type(result)}")
    
    return result


def find_best_alt_candidate(
    responses_state: object,
    target_size: int,
    target_ext: str,
    exclude_usernames: Set[str] = None,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Given the SearchState JSON (with 'responses' list) OR a bare list of responses,
    find the best alt candidate:
      - NOT from any user in exclude_usernames (failed sources for this track)
      - same extension where possible
      - size within DEFAULT_MAX_SIZE_DIFF
      - closest size match preferred
      - free upload slot preferred
      - shorter queue length better
      - higher upload speed better

    Returns (best_within_threshold, closest_any) - two dicts or None.
    - best_within_threshold: best candidate within size threshold (for auto-replace)
    - closest_any: closest candidate by size regardless of threshold (for reporting)
    """
    # Normalise responses_state into a list of response dicts
    if isinstance(responses_state, list):
        responses = responses_state
    elif isinstance(responses_state, dict):
        responses = responses_state.get("responses") or responses_state.get("items") or []
    else:
        responses = []

    best: Optional[dict] = None
    best_score: Optional[Tuple[int, int, int]] = None
    
    # Track closest match regardless of threshold
    closest: Optional[dict] = None
    closest_diff: Optional[float] = None
    
    # Debug counters
    total_files = 0
    filtered_ext = 0
    filtered_size = 0

    for resp in responses:
        # Debug: show what keys are in each response
        if responses and total_files == 0:
            print(f"[ALT-DEBUG] First response keys: {list(resp.keys()) if isinstance(resp, dict) else type(resp)}")
        
        username = resp.get("username") or ""
        
        # Skip any previously failed sources for this track
        if exclude_usernames and username.lower() in {u.lower() for u in exclude_usernames}:
            continue
        
        queue_len = int(resp.get("queueLength") or 0)
        free_slot = bool(resp.get("hasFreeUploadSlot", False))
        upload_speed = int(resp.get("uploadSpeed") or 0)
        files = resp.get("files") or []

        for f in files:
            total_files += 1
            ext = (f.get("extension") or "").lower().lstrip(".")
            # Must match target extension (filter if no ext or different ext)
            if target_ext:
                if not ext or ext != target_ext:
                    filtered_ext += 1
                    continue

            size = int(f.get("size") or 0)
            path = f.get("filename") or f.get("fullname") or ""
            
            # Calculate size diff - reject files with 0 size
            if size <= 0:
                filtered_size += 1
                continue  # Skip 0-byte files entirely
            if target_size > 0:
                rel_diff = abs(size - target_size) / float(target_size)
            else:
                rel_diff = 1.0  # If target is 0, any size is a mismatch
            
            # Track closest match by size (for reporting)
            if closest_diff is None or rel_diff < closest_diff:
                closest_diff = rel_diff
                closest = {
                    "username": username,
                    "queueLength": queue_len,
                    "hasFreeUploadSlot": free_slot,
                    "uploadSpeed": upload_speed,
                    "file": f,
                    "path": path,
                    "size": size,
                    "extension": ext,
                    "size_diff_pct": rel_diff * 100,
                }
            
            # Filter by threshold for auto-replace candidate
            if target_size > 0 and size > 0 and rel_diff > DEFAULT_MAX_SIZE_DIFF:
                filtered_size += 1
                continue

            # Score: prioritize CLOSEST SIZE first, then free slot, queue, speed as tiebreakers
            score = (
                -rel_diff,                 # closest size is best (negative so smaller diff = higher score)
                1 if free_slot else 0,     # then prefer free slots
                -queue_len,                # smaller queue is better
                upload_speed,              # faster better
            )

            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "username": username,
                    "queueLength": queue_len,
                    "hasFreeUploadSlot": free_slot,
                    "uploadSpeed": upload_speed,
                    "file": f,
                    "path": path,
                    "size": size,
                    "extension": ext,
                    "size_diff_pct": rel_diff * 100,  # Include for logging
                }

    # Debug output
    print(f"[ALT-DEBUG] responses={len(responses)}, total_files={total_files}, "
          f"filtered_ext={filtered_ext}, filtered_size={filtered_size}, "
          f"candidates_evaluated={total_files - filtered_ext - filtered_size}")
    
    return best, closest


def format_size(mi_bytes: float) -> str:
    return f"{mi_bytes:.2f} MiB"


DEFAULT_AUTO_REPLACE_THRESHOLD = 5.0  # percent


def enqueue_download(slskd: SlskdSession, username: str, filename: str, size: int) -> Tuple[bool, str]:
    """Enqueue a download from a specific user."""
    try:
        files = [{"filename": filename, "size": size}]
        resp = slskd.post(f"/transfers/downloads/{quote(username)}", json=files)
        if resp.status_code < 400:
            return True, "ok"
        return False, f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


def cancel_download(slskd: SlskdSession, item: DownloadItem, remove: bool = True) -> Tuple[bool, str]:
    """Cancel (and optionally remove) a stuck download."""
    try:
        params = {"remove": "true"} if remove else {}
        resp = slskd.delete(f"/transfers/downloads/{quote(item.key.username)}/{quote(item.id)}", params=params)
        if resp.status_code < 400:
            return True, "ok"
        return False, f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


def prompt_alt_replacement(
    slskd: SlskdSession,
    item: DownloadItem,
    alt: dict,
    auto_mode: bool = False,
    auto_replace: bool = False,
    auto_replace_threshold: float = DEFAULT_AUTO_REPLACE_THRESHOLD,
) -> bool:
    """
    Print a detailed comparison and optionally auto-replace or prompt the user.
    Returns True if replacement was made.
    """
    old_size_mib = item.size / (1024.0 * 1024.0) if item.size else 0.0
    new_size_mib = alt["size"] / (1024.0 * 1024.0) if alt["size"] else 0.0
    diff_bytes = alt["size"] - item.size
    diff_mib = diff_bytes / (1024.0 * 1024.0) if item.size else 0.0
    rel_pct = (diff_bytes / float(item.size) * 100.0) if (item.size and alt["size"]) else 0.0
    abs_rel_pct = abs(rel_pct)

    print()
    print(f"[ALT] Proposed alternative source for track '{clean_track_title(item.key.filename)}':")
    print(f"        Current: {item.display_name()}")
    print(f"          Size:  {format_size(old_size_mib)} ({item.size} bytes)")
    print(f"        Source:  {item.key.username}")
    print()
    print(f"        New:     {alt['username']} :: {alt['path']}")
    print(f"          Size:  {format_size(new_size_mib)} ({alt['size']} bytes)")
    print(f"          Diff:  {format_size(diff_mib)} ({rel_pct:+.1f}%)")
    print(f"          Stats: queue={alt['queueLength']}, free_slot={alt['hasFreeUploadSlot']}, upload_speed={alt['uploadSpeed']}")
    print()
    
    # Auto-replace if within threshold
    if auto_replace and abs_rel_pct <= auto_replace_threshold:
        print(f"[ALT] Auto-replacing (size diff {abs_rel_pct:.1f}% <= {auto_replace_threshold}% threshold)...")
        
        # Cancel the old download
        ok_cancel, reason_cancel = cancel_download(slskd, item, remove=True)
        if not ok_cancel:
            print(f"[ALT] Failed to cancel old download: {reason_cancel}")
            return False
        
        # Enqueue the new download
        ok_enqueue, reason_enqueue = enqueue_download(slskd, alt['username'], alt['path'], alt['size'])
        if ok_enqueue:
            print(f"[ALT] ✓ Replaced! Now downloading from {alt['username']}")
            return True
        else:
            print(f"[ALT] Failed to enqueue new download: {reason_enqueue}")
            return False
    
    if auto_mode:
        if auto_replace and abs_rel_pct > auto_replace_threshold:
            print(f"[ALT] ⚠ NEAR-MISS: Size diff {abs_rel_pct:.1f}% > {auto_replace_threshold}% threshold.")
            print(f"[ALT]   → Manual review: {alt['username']} :: {alt['path']}")
        else:
            print("[ALT] Auto mode: alternative logged. Swap in slskd/Lidarr UI if desired.")
    else:
        try:
            ans = input("Open this candidate in the UI and replace the stuck download there? [y/N]: ").strip().lower()
            if ans == "y":
                print("[ALT] A better source exists; please swap it in slskd and/or Lidarr as needed.")
            else:
                print("[ALT] Keeping current source; script will continue retrying as configured.")
        except EOFError:
            print("[ALT] Non-interactive mode: alternative logged.")
    print()
    return False


# ---------------------- Main processing loop ---------------------- #

def process_queue(
    slskd: SlskdSession,
    queue: List[DownloadItem],
    retry_before_replace: int,
    per_file_cooldown: float,
    snapshot_refresh_interval: float,
    auto_mode: bool = False,
    auto_replace: bool = False,
    auto_replace_threshold: float = DEFAULT_AUTO_REPLACE_THRESHOLD,
    search_timeout_ms: int = DEFAULT_SEARCH_TIMEOUT_MS,
    batch_size: int = DEFAULT_MAX_CONCURRENT_SEARCHES,
) -> None:
    if not queue:
        print("No problem downloads found; nothing to do.")
        return

    # Deduplicate queue by track_key (keep first occurrence)
    seen_tracks: Set[Tuple[str, str]] = set()
    deduped_queue: List[DownloadItem] = []
    for item in queue:
        tk = item.key.track_key()
        if tk not in seen_tracks:
            seen_tracks.add(tk)
            deduped_queue.append(item)
    if len(deduped_queue) < len(queue):
        print(f"[INFO] Deduplicated queue: {len(queue)} -> {len(deduped_queue)} items")
        queue.clear()
        queue.extend(deduped_queue)

    print(
        f"Initial problem queue length: {len(queue)} "
        f"(states considered: {', '.join(sorted(PROBLEM_STATES))})"
    )
    print(
        f"[CFG] per_file_cooldown={per_file_cooldown}s, "
        f"snapshot_refresh_interval={snapshot_refresh_interval}s, "
        f"retry_before_replace={retry_before_replace}"
    )

    track_retry_counts: Dict[Tuple[str, str], int] = {}
    last_retry_at: Dict[DownloadKey, float] = {}
    alt_offered_for: set[Tuple[str, str]] = set()
    # Track failed sources per track - don't cycle back to users we've already tried
    failed_sources_for_track: Dict[Tuple[str, str], Set[str]] = {}
    MAX_FAILED_SOURCES_BEFORE_RESET = 10  # After trying 10 different sources, allow cycling back
    pending_alt_searches: List[Tuple[DownloadItem, str, Tuple[str, str]]] = []  # (item, search_text, track_key)
    last_snapshot_ts = time.monotonic()

    def get_item_priority(item: DownloadItem) -> Tuple[int, float, int]:
        """
        Priority tuple (lower = higher priority):
        1. Retry count (0 retries first)
        2. Time since last attempt (longer ago = higher priority, so negate)
        3. Number of failed sources tried
        """
        tk = item.key.track_key()
        retry_count = track_retry_counts.get(tk, 0)
        last_attempt = last_retry_at.get(item.key, 0)
        time_since_last = now - last_attempt if last_attempt else float('inf')
        failed_source_count = len(failed_sources_for_track.get(tk, set()))
        # Return tuple: lower values = higher priority
        # Negate time_since_last so longer waits get priority
        return (retry_count, -time_since_last, failed_source_count)
    
    def sort_queue_by_priority():
        """Sort queue so fresh items and long-waiting items come first."""
        nonlocal queue
        queue.sort(key=get_item_priority)
    
    # Sort initially and process in priority order
    now = time.monotonic()
    sort_queue_by_priority()
    idx = 0
    while idx < len(queue):
        now = time.monotonic()

        # Periodically refresh snapshot and drop items that no longer need retry
        if now - last_snapshot_ts >= snapshot_refresh_interval:
            try:
                raw = fetch_all_downloads(slskd)
                fresh_queue = build_problem_queue(raw)
                fresh_keys = {item.key for item in fresh_queue}
                # Filter existing queue to keys still present
                queue = [item for item in queue if item.key in fresh_keys]
                # Append any newly-added problem items
                existing_keys = {item.key for item in queue}
                for item in fresh_queue:
                    if item.key not in existing_keys:
                        queue.append(item)
                last_snapshot_ts = now
                # Clear alt_offered_for so we re-search (users may have come online)
                alt_offered_for.clear()
                sort_queue_by_priority()  # Re-sort so fresh items come first
                idx = 0  # start from front again
                print(f"[INFO] Snapshot refreshed. Queue now has {len(queue)} items (sorted by priority). Alt-search cache cleared.")
                if not queue:
                    print("[INFO] No remaining problem downloads after refresh.")
                    return
            except Exception as exc:
                print(f"[WARN] Failed to refresh snapshot: {exc}")

        if not queue:
            break

        if idx >= len(queue):
            idx = 0

        item = queue[idx]
        track_key = item.key.track_key()

        # Enforce per-file cooldown
        last_ts = last_retry_at.get(item.key)
        if last_ts is not None and (now - last_ts) < per_file_cooldown:
            # Move this item to the back and continue
            queue.append(queue.pop(idx))
            continue

        # When retry_before_replace=0, skip re-enqueue and go straight to alt search
        # Also skip if we already searched and found no suitable alt
        if retry_before_replace == 0 and track_key not in alt_offered_for:
            print(f"[INFO] Skipping re-enqueue for {item.display_name()} (retry_before_replace=0)")
            ok, reason = False, "skipped"
        elif track_key in alt_offered_for:
            # Already searched recently - move to back for later retry (users may come online)
            # Don't remove from queue - just deprioritize and continue to next item
            queue.append(queue.pop(idx))
            continue
        else:
            ok, reason = reenqueue_download(slskd, item)
        
        last_retry_at[item.key] = time.monotonic()

        # Update retry count for this track (count only failed cycles?)
        track_retry_counts[track_key] = track_retry_counts.get(track_key, 0) + (0 if ok else 1)

        if ok:
            print(f"[OK] Re-enqueued {item.display_name()} (state={item.state})")
            # Keep in queue but move to back - only remove when actually REPLACED
            queue.append(queue.pop(idx))
            continue

        # Not OK
        if reason == "offline":
            print(f"[WARN] Peer offline for {item.display_name()}")
        elif reason != "skipped":
            print(f"[WARN] Failed to re-enqueue {item.display_name()}: {reason}")

        # Collect items for batched alt-source search
        needs_alt_search = (
            track_retry_counts[track_key] >= retry_before_replace
            and track_key not in alt_offered_for
        )
        
        if needs_alt_search:
            # Add current failing source to the blacklist for this track
            if track_key not in failed_sources_for_track:
                failed_sources_for_track[track_key] = set()
            failed_sources_for_track[track_key].add(item.key.username)
            
            # Reset blacklist if we've tried too many sources (allow cycling back)
            if len(failed_sources_for_track[track_key]) >= MAX_FAILED_SOURCES_BEFORE_RESET:
                print(f"[INFO] Tried {len(failed_sources_for_track[track_key])} sources for '{track_key[1]}', resetting blacklist")
                failed_sources_for_track[track_key] = {item.key.username}  # Keep only current
            
            # Collect this item for batch processing
            search_text = build_search_text(item)
            pending_alt_searches.append((item, search_text, track_key))
            alt_offered_for.add(track_key)
            
            # Process batch when full or at end of queue
            if len(pending_alt_searches) >= batch_size or idx >= len(queue) - 1:
                print(f"\n[SEARCH] Processing {len(pending_alt_searches)} alt-source searches...")
                
                # Run sliding window search (keeps max_concurrent searches running at all times)
                search_results = sliding_window_search(
                    slskd,
                    [(itm, stxt) for itm, stxt, _ in pending_alt_searches],
                    max_concurrent=batch_size,
                    search_timeout_ms=search_timeout_ms,
                )
                
                # Process results
                items_to_remove = set()
                for batch_item, batch_search_text, batch_track_key in pending_alt_searches:
                    responses = search_results.get(batch_item)
                    track_name = clean_track_title(batch_item.key.filename)
                    
                    if responses is None:
                        print(f"[ALT] Search failed/timed out for '{track_name}' (search: '{batch_search_text}')")
                    elif len(responses) == 0:
                        print(f"[ALT] Zero results on network for '{track_name}' (search: '{batch_search_text}')")
                    else:
                        # Exclude all previously failed sources for this track
                        excluded = failed_sources_for_track.get(batch_track_key, set())
                        if excluded:
                            print(f"[ALT-DEBUG] Excluding {len(excluded)} previously tried sources: {', '.join(sorted(excluded))}")
                        best, closest = find_best_alt_candidate(
                            responses, batch_item.size, batch_item.ext,
                            exclude_usernames=excluded
                        )
                        if not best:
                            if closest:
                                print(f"[ALT] No auto-replace candidate for '{track_name}'")
                                print(f"[ALT]   → Closest match: {closest['size_diff_pct']:.1f}% diff - {closest['username']} :: {closest['path']}")
                            else:
                                print(f"[ALT] No suitable alt for '{track_name}'")
                        else:
                            replaced = prompt_alt_replacement(
                                slskd, batch_item, best,
                                auto_mode=auto_mode,
                                auto_replace=auto_replace,
                                auto_replace_threshold=auto_replace_threshold,
                            )
                            if replaced:
                                items_to_remove.add(batch_track_key)
                
                # Remove replaced items from queue
                if items_to_remove:
                    orig_len = len(queue)
                    queue[:] = [q for q in queue if q.key.track_key() not in items_to_remove]
                    print(f"[BATCH] Removed {orig_len - len(queue)} replaced item(s) from queue")
                    idx = 0  # Restart from beginning after batch removal
                
                pending_alt_searches.clear()

        # Move this item to the back of the queue
        queue.append(queue.pop(idx))

    print("Done for this run.")


# ---------------------- CLI ---------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry stuck slskd downloads and suggest alternatives."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SLSKD_BASE_URL", DEFAULT_BASE_URL),
        help=f"Base URL for slskd API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SLSKD_API_KEY"),
        help="slskd API key (or set SLSKD_API_KEY env var)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=f"HTTP request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT})",
    )
    parser.add_argument(
        "--per-file-cooldown",
        type=float,
        default=DEFAULT_PER_FILE_COOLDOWN,
        help=f"Minimum seconds between retries of the same track (default: {DEFAULT_PER_FILE_COOLDOWN})",
    )
    parser.add_argument(
        "--snapshot-refresh-interval",
        type=float,
        default=DEFAULT_SNAPSHOT_REFRESH_INTERVAL,
        help=f"Seconds between snapshot refreshes from slskd (default: {DEFAULT_SNAPSHOT_REFRESH_INTERVAL})",
    )
    parser.add_argument(
        "--retry-before-replace",
        type=int,
        default=DEFAULT_RETRY_BEFORE_REPLACE,
        help=(
            "Number of failed retry cycles for a track before we try to find an alternative "
            f"source (default: {DEFAULT_RETRY_BEFORE_REPLACE})"
        ),
    )
    parser.add_argument(
        "--print-states",
        action="store_true",
        help="Print the set of observed transfer states and exit.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run in non-interactive mode (no prompts, just log alternatives).",
    )
    parser.add_argument(
        "--auto-replace",
        action="store_true",
        help="Automatically replace stuck downloads with alternatives (implies --auto).",
    )
    parser.add_argument(
        "--auto-replace-threshold",
        type=float,
        default=DEFAULT_AUTO_REPLACE_THRESHOLD,
        help=f"Max size difference %% for auto-replace (default: {DEFAULT_AUTO_REPLACE_THRESHOLD}%%).",
    )
    parser.add_argument(
        "--search-timeout",
        type=int,
        default=DEFAULT_SEARCH_TIMEOUT_MS,
        help=f"Search timeout in ms (default: {DEFAULT_SEARCH_TIMEOUT_MS}ms). Lower = faster but fewer results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_SEARCHES,
        help=f"Number of parallel searches to run (default: {DEFAULT_MAX_CONCURRENT_SEARCHES}, max ~10).",
    )
    parser.add_argument(
        "--find",
        type=str,
        default=None,
        help="Search for a specific string in all download filenames (case-insensitive) and show their states.",
    )

    args = parser.parse_args(argv)
    
    # --auto-replace implies --auto
    if args.auto_replace:
        args.auto = True

    if not args.api_key:
        parser.error("You must provide an API key via --api-key or SLSKD_API_KEY env var.")

    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    slskd = SlskdSession(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.request_timeout,
    )

    if args.print_states:
        try:
            states = collect_states(slskd)
        except Exception as exc:
            print(f"ERROR: Failed to query downloads from slskd: {exc}")
            sys.exit(1)
        print("Observed transfer states:")
        for st in states:
            print(f"  {st}")
        return

    try:
        raw = fetch_all_downloads(slskd)
    except Exception as exc:
        print(f"ERROR: Failed to query downloads from slskd: {exc}")
        sys.exit(1)

    # Handle --find: search for specific filename
    if args.find:
        search_term = args.find.lower()
        print(f"Searching for '{args.find}' in all downloads...\n")
        found_count = 0
        for item in iter_download_items(raw):
            if search_term in item.key.filename.lower() or search_term in item.key.directory.lower():
                in_problem = "✓ PROBLEM" if item.state in PROBLEM_STATES else "✗ NOT PROBLEM"
                print(f"[{in_problem}] state={item.state}")
                print(f"  user: {item.key.username}")
                print(f"  dir:  {item.key.directory}")
                print(f"  file: {item.key.filename}")
                print(f"  size: {item.size / 1024 / 1024:.2f} MiB")
                print()
                found_count += 1
        if found_count == 0:
            print(f"No downloads found matching '{args.find}'")
        else:
            print(f"Found {found_count} download(s) matching '{args.find}'")
        return

    queue = build_problem_queue(raw, debug=True)
    process_queue(
        slskd=slskd,
        queue=queue,
        retry_before_replace=args.retry_before_replace,
        per_file_cooldown=args.per_file_cooldown,
        snapshot_refresh_interval=args.snapshot_refresh_interval,
        auto_mode=args.auto,
        auto_replace=args.auto_replace,
        auto_replace_threshold=args.auto_replace_threshold,
        search_timeout_ms=args.search_timeout,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()




