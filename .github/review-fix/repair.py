from pathlib import Path

path = Path('.github/review-fix/apply.py')
text = path.read_text()

old = 'range.start <= range.end && (range.end as usize) <= self.len(),'
new = 'range.start <= range.end && range.end as usize <= self.len(),'
if text.count(old) != 1:
    raise RuntimeError(f'expected one range transform anchor, found {text.count(old)}')
text = text.replace(old, new, 1)

old = '            || !self.ctx_other.is_null()\n'
if text.count(old) != 1:
    raise RuntimeError(f'expected one ctx_other transform line, found {text.count(old)}')
text = text.replace(old, '', 1)

path.write_text(text)
