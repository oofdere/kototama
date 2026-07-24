from pathlib import Path

path = Path('.github/review-fix/apply.py')
text = path.read_text()
old = '''    fn checked_copy_range(&self, range: Range<usize>) -> Range<i32> {
        let range = Self::checked_range(range).expect("sequence range exceeds llama_pos");
        assert!(
            range.start <= range.end && (range.end as usize) <= self.len(),
            "sequence range out of bounds"
        );
        range
    }'''
new = '''    fn checked_copy_range(&self, range: Range<usize>) -> Range<i32> {
        assert!(
            range.start <= range.end && range.end <= self.len(),
            "sequence range out of bounds"
        );
        Self::checked_range(range).expect("sequence range exceeds llama_pos")
    }'''
if text.count(old) != 1:
    raise RuntimeError(f'expected one transform anchor, found {text.count(old)}')
path.write_text(text.replace(old, new, 1))
