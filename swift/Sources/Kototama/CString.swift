//
//  CString.swift
//  Kototama
//
//  One place for converting C strings to Swift text.
//

import Foundation

/// Decodes a null-terminated C string lossily.
///
/// llama.cpp hands back token pieces and vocabulary text as raw bytes, and
/// byte-fallback tokens are *not* valid UTF-8 on their own. Swift's
/// `String(cString:)` traps on invalid UTF-8, so we copy through `UInt8` and
/// use `String(decoding:as:)`, which inserts U+FFFD for bad bytes — the same
/// never-fails behavior as Rust's `String::from_utf8_lossy`.
func lossyString(_ cString: UnsafePointer<CChar>) -> String {
    var bytes: [UInt8] = []
    var pointer = UnsafeRawPointer(cString).assumingMemoryBound(to: UInt8.self)
    while pointer.pointee != 0 {
        bytes.append(pointer.pointee)
        pointer = pointer.advanced(by: 1)
    }
    return String(decoding: bytes, as: UTF8.self)
}

/// Decodes a bounded run of C bytes lossily, stopping at the first NUL.
///
/// Use when the C API reports how many bytes it wrote into a caller-supplied
/// buffer that is zero-padded past that point.
func lossyString(_ buffer: ArraySlice<CChar>, count: Int) -> String {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(count)
    for value in buffer.prefix(count) {
        let byte = UInt8(bitPattern: value)
        if byte == 0 { break }
        bytes.append(byte)
    }
    return String(decoding: bytes, as: UTF8.self)
}
