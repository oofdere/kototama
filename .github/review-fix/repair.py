from pathlib import Path

path = Path('.github/review-fix/apply.py')
text = path.read_text()
old = 'range.start <= range.end && (range.end as usize) <= self.len(),'
new = 'range.start <= range.end && range.end as usize <= self.len(),'
if text.count(old) != 1:
    raise RuntimeError(f'expected one transform anchor, found {text.count(old)}')
path.write_text(text.replace(old, new, 1))
