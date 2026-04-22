import re
import csv
import ast
import sys
from datetime import datetime
from pathlib import Path

# 추출할 컬럼 순서 고정
TARGET_COLUMNS = [
    "loss",
    "grad_norm",
    "learning_rate",
    "ppl",
    "memory/max_active (GiB)",
    "memory/max_allocated (GiB)",
    "memory/device_reserved (GiB)",
    "tokens/train_per_sec_per_gpu",
    "tokens/total",
    "tokens/trainable",
    "epoch",
]

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S,%f"
START_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\].*Starting trainer\.\.\.")
END_PATTERN   = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\].*Training completed!")


def parse_training_duration(text: str) -> str | None:
    """훈련 시작/종료 타임스탬프로 총 소요시간을 계산합니다."""
    start_match = START_PATTERN.search(text)
    end_match   = END_PATTERN.search(text)

    if not start_match or not end_match:
        return None

    t_start = datetime.strptime(start_match.group(1), TIMESTAMP_FMT)
    t_end   = datetime.strptime(end_match.group(1),   TIMESTAMP_FMT)
    delta   = t_end - t_start

    total_sec = int(delta.total_seconds())
    h, rem    = divmod(total_sec, 3600)
    m, s      = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_benchmark_log(text: str) -> list[dict]:
    """로그에서 지정된 컬럼만 추출합니다."""
    # "Starting trainer..." 이후 텍스트만 파싱 (앞의 config JSON 제외)
    marker = re.search(r"\[.*?\] \[INFO\] \[axolotl\.train\] Starting trainer\.\.\.", text)
    text = text[marker.end():] if marker else text

    pattern = r"\{[^{}]+\}"
    records = []

    for match in re.findall(pattern, text):
        try:
            raw = ast.literal_eval(match)
            # train_runtime이 포함된 최종 요약 레코드는 제외
            if "train_runtime" in raw:
                continue
            # 학습 레코드가 아닌 경우 제외 (loss 키가 없으면 skip)
            if "loss" not in raw:
                continue
            # TARGET_COLUMNS에 해당하는 키만 추출 (없는 키는 빈 문자열)
            record = {col: raw.get(col, "") for col in TARGET_COLUMNS}
            records.append(record)
        except (ValueError, SyntaxError):
            print(f"[경고] 파싱 실패: {match[:80]}...")

    return records


def save_to_csv(records: list[dict], duration: str | None, output_path: str):
    """데이터를 CSV로 저장하고, 첫 번째 행에 총 소요시간을 기록합니다."""
    if not records:
        print("저장할 데이터가 없습니다.")
        return

    fieldnames = TARGET_COLUMNS + ["total_duration"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, record in enumerate(records):
            # 첫 번째 레코드에만 total_duration 기록
            total_duration = duration if (i == 0 and duration) else ""
            writer.writerow({**record, "total_duration": total_duration})

    print(f"✅ {len(records)}개 레코드를 '{output_path}'에 저장했습니다.")
    if duration:
        print(f"   총 소요시간: {duration}")
    else:
        print("   ⚠️  시작/종료 타임스탬프를 찾지 못해 소요시간을 계산하지 못했습니다.")


def process_file(log_path: Path):
    log_text = log_path.read_text(encoding="utf-8")
    records  = parse_benchmark_log(log_text)
    duration = parse_training_duration(log_text)
    csv_path = log_path.with_suffix(".csv")
    print(f"\n📄 {log_path.name} → {csv_path.name} ({len(records)}개 레코드)")
    save_to_csv(records, duration, str(csv_path))


def main():
    if len(sys.argv) < 2:
        print("사용법: python parse_benchmark.py <폴더경로>")
        print("예시:   python parse_benchmark.py ./logs")
        sys.exit(1)

    target = Path(sys.argv[1])

    if not target.exists():
        print(f"경로를 찾을 수 없습니다: {target}")
        sys.exit(1)

    log_files = sorted(target.glob("*.txt"))

    if not log_files:
        print(f"'{target}' 폴더에 .txt 파일이 없습니다.")
        sys.exit(1)

    print(f"🔍 {len(log_files)}개의 .txt 파일을 발견했습니다.")
    for log_path in log_files:
        process_file(log_path)

    print(f"\n✅ 완료: {len(log_files)}개 파일 변환")


if __name__ == "__main__":
    main()