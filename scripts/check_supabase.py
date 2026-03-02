#!/usr/bin/env python3
"""
ApexHydra — Supabase connectivity and schema check.
Run after deploying schema (or migration). Requires SUPABASE_URL and SUPABASE_KEY in env.
  python scripts/check_supabase.py
"""
import os
import sys

def main():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_KEY (e.g. in .env or export).")
        sys.exit(1)

    try:
        from supabase import create_client
    except ImportError:
        print("Install: pip install supabase")
        sys.exit(1)

    sb = create_client(url, key)
    errors = []

    # ea_config: need magic and at least one row for EA; check for live_* columns
    try:
        r = sb.table("ea_config").select("*").limit(1).execute()
        if r.data and len(r.data) > 0:
            row = r.data[0]
            has_live = "live_balance" in row or "live_ts" in row
            has_alloc = "allocated_capital" in row
            print("ea_config: OK (row exists)" + ("" if has_live else " — consider running migration for live_*"))
            if not has_alloc:
                errors.append("ea_config missing allocated_capital")
        else:
            print("ea_config: OK (empty — EA will create via first PATCH or use default)")
    except Exception as e:
        errors.append(f"ea_config: {e}")

    # trades
    try:
        r = sb.table("trades").select("id,symbol,action").limit(1).execute()
        print("trades: OK")
    except Exception as e:
        errors.append(f"trades: {e}")

    # performance
    try:
        r = sb.table("performance").select("id,balance,equity").limit(1).execute()
        print("performance: OK")
    except Exception as e:
        errors.append(f"performance: {e}")

    # events
    try:
        r = sb.table("events").select("id,type").limit(1).execute()
        print("events: OK")
    except Exception as e:
        errors.append(f"events: {e}")

    if errors:
        print("\nIssues:", *errors, sep="\n  - ")
        sys.exit(1)
    print("\nSupabase check passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
