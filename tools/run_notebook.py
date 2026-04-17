import argparse
import json
import os
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    args = parser.parse_args()

    nb_path = Path(args.notebook).resolve()
    os.chdir(nb_path.parent)
    os.environ.setdefault("MPLBACKEND", "Agg")

    with nb_path.open(encoding="utf-8") as f:
        data = json.load(f)

    ns = {"__name__": "__main__", "__file__": str(nb_path)}

    for idx, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        print(f"\n===== Running code cell {idx} =====")
        try:
            exec(compile(source, f"{nb_path.name}#cell{idx}", "exec"), ns, ns)
        except Exception:
            print(f"\nCell {idx} failed.")
            traceback.print_exc()
            return 1

    print(f"\nNotebook completed successfully: {nb_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
