//
//  CValueBox.swift
//  Kototama
//
//  A tiny wrapper that lets C structs live inside Swift value types.
//
//  Swift only allows stored properties in `Sendable` structs when their types
//  are themselves `Sendable`, and imported C structs are not (they may contain
//  pointers). Wrapping one in a class marked `@unchecked Sendable` is the
//  standard escape hatch: the box owns plain data that we never mutate
//  concurrently, so the annotation is sound in practice.
//

/// Wraps a C struct so it can be stored in a `Sendable` Swift type.
final class CValueBox<T>: @unchecked Sendable {
    /// The wrapped C value. Mutable because callers adjust fields in place
    /// before passing the box down to C (e.g. `ModelParams.nGPULayers = 9`).
    var value: T

    init(_ value: T) {
        self.value = value
    }
}
