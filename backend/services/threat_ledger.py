# backend/services/threat_ledger.py

import os
import io
import csv
import time
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
import aiofiles

logger = logging.getLogger("LinguaShield.ThreatLedger")

LEDGER_FILE = "threat_ledger.csv"
LEDGER_HEADER = [
    "session_id",
    "timestamp",
    "tab",
    "phase",
    "payload",
    "asset_id",
    "generation",
]

_ledger_lock = asyncio.Lock()
ASSET_SCOPED_PHASES = {"INPUT", "OUTPUT", "CHAT_USER", "CHAT_BOT"}


async def initialize_ledger() -> None:
    async with _ledger_lock:
        if os.path.exists(LEDGER_FILE):
            async with aiofiles.open(LEDGER_FILE, mode="r", encoding="utf-8") as f:
                first_line = await f.readline()
            if first_line and "generation" not in first_line:
                async with aiofiles.open(LEDGER_FILE, mode="r", encoding="utf-8") as f:
                    content = await f.read()
                reader = csv.reader(io.StringIO(content))
                rows = list(reader)
                new_rows = [LEDGER_HEADER]
                for row in rows[1:]:
                    if len(row) < len(LEDGER_HEADER):
                        row.append("")
                    new_rows.append(row)
                out_buf = io.StringIO()
                writer = csv.writer(out_buf, quoting=csv.QUOTE_ALL)
                writer.writerows(new_rows)
                async with aiofiles.open(LEDGER_FILE, mode="w", encoding="utf-8", newline="") as f:
                    await f.write(out_buf.getvalue())
            return

        async with aiofiles.open(
            LEDGER_FILE,
            mode="w",
            encoding="utf-8",
            newline="",
        ) as f:
            buffer = io.StringIO()
            csv.writer(buffer).writerow(LEDGER_HEADER)
            await f.write(buffer.getvalue())


async def log_ledger_event(
    session_id: str,
    tab: str,
    phase: str,
    payload: Any,
    asset_id: Optional[str] = None,
    generation: Optional[int] = None,
) -> None:
    if not session_id:
        raise ValueError("session_id is required")

    if phase in ASSET_SCOPED_PHASES:
        if not asset_id:
            raise ValueError(f"Phase '{phase}' strictly requires a valid asset_id.")
        if generation is None:
            raise ValueError(f"Phase '{phase}' strictly requires a valid generation.")

    await initialize_ledger()
    timestamp = time.time()

    if isinstance(payload, (dict, list)):
        payload_str = json.dumps(payload, ensure_ascii=False)
    else:
        payload_str = str(payload)

    row_buffer = io.StringIO()
    csv.writer(row_buffer, quoting=csv.QUOTE_ALL).writerow(
        [
            session_id,
            timestamp,
            tab,
            phase,
            payload_str,
            asset_id or "",
            str(generation) if generation is not None else "",
        ]
    )

    async with _ledger_lock:
        async with aiofiles.open(
            LEDGER_FILE,
            mode="a",
            encoding="utf-8",
            newline="",
        ) as f:
            await f.write(row_buffer.getvalue())


async def get_session_history_from_ledger(
    session_id: str,
) -> List[Dict[str, Any]]:
    if not os.path.exists(LEDGER_FILE):
        return []

    async with _ledger_lock:
        async with aiofiles.open(
            LEDGER_FILE,
            mode="r",
            encoding="utf-8",
            newline="",
        ) as f:
            content = await f.read()

    reader = csv.reader(io.StringIO(content))
    header = next(reader, None)
    records: List[Dict[str, Any]] = []

    for row in reader:
        if len(row) < 5:
            continue
        if row[0] != session_id:
            continue

        try:
            timestamp = float(row[1])
        except (TypeError, ValueError):
            continue

        gen_val = None
        if len(row) > 6 and row[6]:
            try:
                gen_val = int(row[6])
            except ValueError:
                pass

        records.append(
            {
                "session_id": row[0],
                "timestamp": timestamp,
                "tab": row[2],
                "phase": row[3],
                "payload": row[4],
                "asset_id": row[5] if len(row) > 5 and row[5] else None,
                "generation": gen_val,
            }
        )

    return records


def get_records_for_asset(
    records: List[Dict[str, Any]],
    session_id: str,
    asset_id: str,
    generation: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not session_id or not asset_id:
        return []
    
    filtered = []
    for record in records:
        if record.get("session_id") == session_id and record.get("asset_id") == asset_id:
            if generation is not None:
                if record.get("generation") == generation:
                    filtered.append(record)
            else:
                filtered.append(record)
    return filtered


async def verify_asset_io_integrity_async(
    session_id: str,
    asset_id: str,
    generation: int,
) -> bool:
    records = await get_session_history_from_ledger(session_id)
    asset_records = get_records_for_asset(records, session_id, asset_id, generation)
    
    input_records = [r for r in asset_records if r.get("phase") == "INPUT" and r.get("generation") == generation]
    output_records = [r for r in asset_records if r.get("phase") == "OUTPUT" and r.get("generation") == generation]

    if not input_records or not output_records:
        return False

    latest_input = max(input_records, key=lambda r: r.get("timestamp", 0.0))
    latest_output = max(output_records, key=lambda r: r.get("timestamp", 0.0))

    if latest_input.get("generation") != latest_output.get("generation"):
        return False

    return latest_input.get("timestamp", 0.0) < latest_output.get("timestamp", 0.0)