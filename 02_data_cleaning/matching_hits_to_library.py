#python matching_hits_to_library.py --library "path\canonicalized_library.csv" --hits "path\hits_canonicalized.csv" --output "path\hit_match_report.txt"

import argparse
import pandas as pd

LIB_FILE = "canonicalized_library.csv"
HITS_FILE = "hits_canonicalized.csv"
OUTPUT_FILE = "hit_match_report.txt"
SMILES_COL = "Smiles"
NAME_COL = "Name"


def load_and_match_by_smiles(lib_path: str, hits_path: str, output_path: str, smiles_col: str = SMILES_COL, name_col: str = NAME_COL):
	lib = pd.read_csv(lib_path)
	hits = pd.read_csv(hits_path)

	if smiles_col not in lib.columns:
		raise KeyError(f"SMILES column '{smiles_col}' not found in library. Available: {list(lib.columns)}")
	if smiles_col not in hits.columns:
		raise KeyError(f"SMILES column '{smiles_col}' not found in hits. Available: {list(hits.columns)}")
	if name_col not in hits.columns:
		raise KeyError(f"Name column '{name_col}' not found in hits. Available: {list(hits.columns)}")

	# Build map from library SMILES to 1-based row numbers
	lib_map = {}
	for idx, smi in lib[smiles_col].items():
		if pd.notna(smi):
			lib_map.setdefault(smi, []).append(idx + 1)

	lines = []
	found_count = 0
	not_found = []

	for _, row in hits.iterrows():
		smi = row[smiles_col]
		name = row[name_col]
		rows = lib_map.get(smi)
		if rows:
			found_count += 1
			lines.append(f"MATCH: {name} | {smi} -> library rows {', '.join(map(str, rows))}")
		else:
			not_found.append((name, smi))

	# Summary and not found section
	lines.insert(0, f"Total matches: {found_count}")
	if not_found:
		lines.append("")
		lines.append(f"Not found hits ({len(not_found)}):")
		for name, smi in not_found:
			lines.append(f"NOT FOUND: {name} | {smi}")

	with open(output_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))

	print(f"Wrote match report → {output_path}")


def parse_args():
	parser = argparse.ArgumentParser(description="Match hits to library based on SMILES and write a TXT report")
	parser.add_argument("--library", "-l", default=LIB_FILE, help="Canonicalized library CSV (default: canonicalized_library.csv)")
	parser.add_argument("--hits", "-H", dest="hits", default=HITS_FILE, help="Canonicalized hits CSV (default: hits_canonicalized.csv)")
	parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="Output TXT report path (default: hit_match_report.txt)")
	parser.add_argument("--name-col", "-n", default=NAME_COL, help="Name column in hits (default: name)")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	load_and_match_by_smiles(args.library, args.hits, args.output, SMILES_COL, args.name_col)

