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
    let _b = Backend::acquire();
    drop(_b);
    let _b2 = Backend::acquire();
}

#[test]
fn concurrent_acquire_is_safe() {
    // Stress test: many threads racing to acquire Backend handles.
    // Before the fix (AtomicUsize ref-counting), this could trigger a race
    // where one thread frees the backend while another re-initializes it.
    // With std::sync::Once, initialization is synchronized and happens exactly
    // once, eliminating the race.
    let handles: Vec<_> = (0..16)
        .map(|_| {
            std::thread::spawn(|| {
                let b = Backend::acquire();
                std::hint::black_box(&b);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
