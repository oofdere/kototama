use rusty_llama::Backend;

#[test]
fn init() {
    let _b = Backend::acquire();
}

#[test]
fn acquire_twice_coexist() {
    let a = Backend::acquire();
    let b = Backend::acquire();
    // Both alive — just confirm no crash
    drop(a);
    drop(b);
}

#[test]
fn acquire_drop_acquire() {
    // Dropping all handles and re-acquiring should re-init cleanly
    {
        let _b = Backend::acquire();
    }
    let _b2 = Backend::acquire();
}

#[test]
fn concurrent_acquire_drop_does_not_race() {
    // Regression test for the TOCTOU race between an atomic refcount transition
    // (0→1 / 1→0) and the corresponding `llama_backend_init` / `llama_backend_free`
    // FFI call. Pre-fix, two threads could observe complementary transitions
    // and run init concurrently with free (or have one thread start using the
    // backend before another finished initializing it). With the mutex-guarded
    // implementation, every transition is serialized with its FFI side-effect,
    // so this stress loop must complete cleanly.
    use std::thread;
    let threads: Vec<_> = (0..8)
        .map(|_| {
            thread::spawn(|| {
                for _ in 0..50 {
                    let _b = Backend::acquire();
                    // drops at end of scope
                }
            })
        })
        .collect();
    for t in threads {
        t.join().expect("worker thread panicked");
    }
}
