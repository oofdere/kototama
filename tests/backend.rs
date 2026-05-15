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
