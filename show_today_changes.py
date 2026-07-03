import os
import datetime

today = datetime.date.today()
project_root = "."

print(f"오늘({today}) 수정된 파일 목록")
print("=" * 60)

changed = []
for root, dirs, files in os.walk(project_root):
    # 불필요한 폴더 제외
    dirs[:] = [d for d in dirs if d not in [
        '.venv', '__pycache__', '.git', 'node_modules',
        '.streamlit', 'htmlcov'
    ]]
    for file in files:
        filepath = os.path.join(root, file)
        try:
            mtime = os.path.getmtime(filepath)
            mdate = datetime.date.fromtimestamp(mtime)
            if mdate == today:
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
                size = os.path.getsize(filepath)
                changed.append((mtime, mtime_str, filepath, size))
        except Exception:
            continue

# 수정 시간 순 정렬
changed.sort(key=lambda x: x[0])

if not changed:
    print("오늘 수정된 파일이 없습니다.")
else:
    for _, mtime_str, filepath, size in changed:
        clean_path = filepath.replace(".\\", "").replace("./", "")
        print(f"[{mtime_str}]  {clean_path}  ({size:,} bytes)")

print()
print(f"총 {len(changed)}개 파일")
