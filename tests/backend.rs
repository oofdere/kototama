use rusty_llama::test_common::load_model;
use rusty_llama::Backend;
use std::sync::{Arc, Barrier};

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
fn concurrent_acquire_and_release() {
    // Acquiring and releasing from several threads at once must not tear the
    // global backend down while another thread is initializing it or loading a
    // model: init/free and the handle count have to move together.
    const THREADS: usize = 8;
    const ITERS: usize = 64;

    let barrier = Arc::new(Barrier::new(THREADS));
    let workers: Vec<_> = (0..THREADS)
        .map(|i| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..ITERS {
                    let backend = Backend::acquire();
                    if i == 0 {
                        // Touch llama.cpp while other threads churn handles.
                        let model = load_model();
                        assert!(!model.desc().is_empty());
                    }
                    drop(backend);
                }
            })
        })
        .collect();

    for worker in workers {
        worker.join().expect("worker thread panicked");
    }

    // The backend is still usable afterwards.
    let model = load_model();
    assert!(!model.tokenize("hello", true, false).is_empty());
}

#[test]
fn model_keeps_backend_alive() {
    // A live Model owns a handle, so unrelated acquire/release cycles must not
    // free the backend underneath it.
    let model = load_model();
    for _ in 0..4 {
        let _b = Backend::acquire();
    }
    assert!(!model.desc().is_empty());
}
