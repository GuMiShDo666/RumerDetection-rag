import csv
from pathlib import Path

import config


LABEL_NAMES = {
    "0": "非谣言",
    "1": "谣言",
}

NATURAL_VERDICTS = {
    "0": "不是谣言",
    "1": "是谣言",
}

SPLIT_FILES = (
    ("train", config.TRAIN_CSV),
    ("valid", config.VALID_CSV),
    ("test", config.TEST_CSV),
)


def normalize_label(value):
    label = str(value).strip()
    if label not in LABEL_NAMES:
        raise ValueError(f"Unsupported label '{value}'. Expected 0 or 1.")
    return label


def read_split_rows():
    rows = []
    for split_name, csv_path in SPLIT_FILES:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing source dataset: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError(f"{path} must contain text and label columns.")

            for row in reader:
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                label = normalize_label(row.get("label", ""))
                statement = f"{text}{NATURAL_VERDICTS[label]}"
                rows.append(
                    {
                        "id": f"RD-{len(rows) + 1:05d}",
                        "split": split_name,
                        "text": text,
                        "statement": statement,
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                    }
                )
    return rows


def write_merged_csv(rows, output_path=config.RUMOR_DATABASE_CSV):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "statement", "label", "label_name"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                "id": row["id"],
                "statement": row["statement"],
                "label": row["label"],
                "label_name": row["label_name"],
            }
            for row in rows
        )
    return output_path


def row_to_markdown(row):
    return "\n".join(
        [
            f"## Case {row['id']}",
            row["statement"],
            f"Record ID: {row['id']}",
            f"Label: {row['label']}",
            f"Verdict: {row['label_name']}",
            f"Claim: {row['text']}",
        ]
    )


def write_rag_markdown(rows, output_path=config.RUMOR_DATABASE_MARKDOWN):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {label_name: 0 for label_name in LABEL_NAMES.values()}
    for row in rows:
        counts[row["label_name"]] += 1

    sections = [
        "# Rumor Detection RAG Database",
        "This knowledge base stores each labeled claim as a natural-language rumor judgment.",
        "Label meaning: `1` = 谣言, `0` = 非谣言.",
        f"Total records: {len(rows)}",
        f"Rumor records: {counts['谣言']}",
        f"Non-rumor records: {counts['非谣言']}",
        "# Cases",
    ]
    sections.extend(row_to_markdown(row) for row in rows)
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return output_path


def build_rumor_database():
    rows = read_split_rows()
    csv_path = write_merged_csv(rows)
    markdown_path = write_rag_markdown(rows)
    counts = {label_name: 0 for label_name in LABEL_NAMES.values()}
    split_counts = {}
    for row in rows:
        counts[row["label_name"]] += 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    return {
        "rows": len(rows),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
        "label_counts": counts,
        "split_counts": split_counts,
    }


def database_summary():
    csv_path = Path(config.RUMOR_DATABASE_CSV)
    if not csv_path.exists():
        return "Rumor database has not been built yet."

    total = 0
    label_counts = {label_name: 0 for label_name in LABEL_NAMES.values()}
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += 1
            label_counts[row["label_name"]] = label_counts.get(row["label_name"], 0) + 1

    return (
        f"Rows: {total}\n"
        f"Labels: 谣言={label_counts.get('谣言', 0)}, 非谣言={label_counts.get('非谣言', 0)}\n"
        f"CSV: {csv_path}"
    )
