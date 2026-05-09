import argparse
from pathlib import Path


STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GENERIC_12_LEADS = [f"Lead{i}" for i in range(1, 13)]


def find_header_files(path):
    if path.is_file():
        if path.suffix.lower() != ".hea":
            raise ValueError(f"Expected a .hea file, got: {path}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Path does not exist: {path}")

    return sorted(path.rglob("*.hea"))


def rewrite_header(header_path, dry_run=False):
    lines = header_path.read_text().splitlines()
    if not lines:
        return False, "empty header"

    try:
        num_signals = int(lines[0].split()[1])
    except (IndexError, ValueError):
        return False, "could not read signal count"

    if num_signals != 12:
        return False, f"expected 12 signals, found {num_signals}"

    if len(lines) < num_signals + 1:
        return False, "header has fewer signal lines than expected"

    current_leads = [lines[i].split()[-1] for i in range(1, num_signals + 1)]
    if current_leads == STANDARD_12_LEADS:
        return False, "already standard"

    if current_leads != GENERIC_12_LEADS:
        return False, f"unexpected lead names: {', '.join(current_leads)}"

    updated_lines = lines[:]
    for i, lead_name in enumerate(STANDARD_12_LEADS, start=1):
        fields = updated_lines[i].split()
        fields[-1] = lead_name
        updated_lines[i] = " ".join(fields)

    if not dry_run:
        header_path.write_text("\n".join(updated_lines) + "\n")

    return True, "renamed Lead1..Lead12 to standard 12-lead names"


def main():
    parser = argparse.ArgumentParser(
        description="Rename WFDB header signal names from Lead1..Lead12 to I, II, III, aVR, aVL, aVF, V1..V6."
    )
    parser.add_argument("path", help="Path to a .hea file or directory containing .hea files")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    args = parser.parse_args()

    header_files = find_header_files(Path(args.path).expanduser())
    changed = 0
    skipped = 0

    for header_file in header_files:
        did_change, message = rewrite_header(header_file, dry_run=args.dry_run)
        if did_change:
            changed += 1
            status = "WOULD UPDATE" if args.dry_run else "UPDATED"
            print(f"{status} {header_file}: {message}")
        else:
            skipped += 1
            print(f"SKIPPED {header_file}: {message}")

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {changed} header(s); skipped {skipped}.")


if __name__ == "__main__":
    main()
