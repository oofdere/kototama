//! `Backend::drop` must not fail when the process' stdout is unusable.
//!
//! The last `Backend` handle is usually dropped from a destructor (`Model`
//! owns one), often during unwinding. If dropping it writes to stdout and the
//! reader of that pipe has gone away (`prog | head -1`), the write fails with
//! `EPIPE`, `println!` panics, and a panic inside a destructor that runs while
//! already panicking aborts the whole process.
//!
//! The scenario is reproduced in a child process (this same test binary): the
//! parent closes its end of the child's stdout pipe right before the child
//! drops its backend handle.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use rusty_llama::Backend;

const CHILD_ENV: &str = "RUSTY_LLAMA_BACKEND_DROP_CHILD";

#[test]
fn child_drops_backend_with_broken_stdout() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }

    let backend = Backend::acquire();

    eprintln!("READY");
    let mut line = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut line)
        .expect("parent should signal that stdout is closed");

    drop(backend);

    // The harness would report the result on stdout, which is now broken.
    std::process::exit(0);
}

#[test]
fn dropping_last_backend_handle_survives_closed_stdout() {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_drops_backend_with_broken_stdout",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn child test process");

    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        let n = stderr.read_line(&mut line).unwrap();
        assert_ne!(n, 0, "child exited before acquiring the backend");
        if line.trim_end() == "READY" {
            break;
        }
    }

    // Close the read end of the child's stdout: its next write gets EPIPE.
    drop(child.stdout.take());

    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(b"go\n").unwrap();
    drop(stdin);

    let status = child.wait().unwrap();
    let mut rest = String::new();
    std::io::Read::read_to_string(&mut stderr, &mut rest).unwrap();

    assert!(
        status.success(),
        "dropping the last Backend handle failed with a broken stdout: {status}\n{rest}"
    );
    assert!(
        !rest.contains("panicked"),
        "child panicked while dropping the Backend:\n{rest}"
    );
}
