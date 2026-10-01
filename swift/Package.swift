// swift-tools-version: 6.0
//
// Kototama for Swift — a human-readable port of the `rusty-llama` Rust crate.
//
// The package links against the same pinned llama.cpp checkout the Rust
// crate uses (../llama-sys/llama.cpp), built as static libraries by
// scripts/build-llama.sh. Run that script once before `swift build`.
//
// Layout:
//   Sources/CLlama        C shim over llama.h (headers copied by the script)
//   Sources/Kototama      the library: Model, Vocabulary, Context, TokenSequence, samplers
//   Sources/simple        example: minimal text generation
//   Sources/SimpleChat    example: interactive chat loop
//   Sources/kototama-bench benchmark harness (mirror of benches/inference.rs)
//   Tests/KototamaTests   test suite (mirror of tests/*.rs)

import PackageDescription
import Foundation

// Absolute paths so the linker finds the prebuilt llama.cpp archives
// regardless of where swift build is invoked from.
let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let llamaLibDir = packageRoot + "/.build/llama/lib"
let llamaArchive = llamaLibDir + "/libkototama-llama.a"

// Everything llama.cpp needs at link time. The static archives are merged
// into one; Swift's linker pulls in the objects whose symbols are referenced,
// exactly as the Rust crate's build.rs does when it links the same archives.
let llamaLinkerSettings: [LinkerSetting] = [
    .unsafeFlags(["-L\(llamaLibDir)"]),
    .linkedLibrary("kototama-llama"),
    .linkedFramework("Metal", .when(platforms: [.macOS])),
    .linkedFramework("Foundation", .when(platforms: [.macOS])),
    .linkedFramework("Accelerate", .when(platforms: [.macOS])),
    .linkedLibrary("c++"),
]

let package = Package(
    name: "kototama",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "Kototama", targets: ["Kototama"]),
        .executable(name: "simple", targets: ["Simple"]),
        .executable(name: "simple-chat", targets: ["SimpleChat"]),
        .executable(name: "kototama-bench", targets: ["KototamaBench"]),
    ],
    targets: [
        // C interop shim. The real headers are copied into include/ by
        // scripts/build-llama.sh; only CLlama.h is ours and checked in.
        .target(
            name: "CLlama",
            path: "Sources/CLlama",
            publicHeadersPath: "include"
        ),
        .target(
            name: "Kototama",
            dependencies: ["CLlama"],
            path: "Sources/Kototama",
            linkerSettings: llamaLinkerSettings
        ),
        .executableTarget(
            name: "Simple",
            dependencies: ["Kototama"],
            path: "Sources/simple"
        ),
        .executableTarget(
            name: "SimpleChat",
            dependencies: ["Kototama"],
            path: "Sources/SimpleChat"
        ),
        .executableTarget(
            name: "KototamaBench",
            dependencies: ["Kototama"],
            path: "Sources/kototama-bench"
        ),
        .testTarget(
            name: "KototamaTests",
            dependencies: ["Kototama"],
            path: "Tests/KototamaTests",
            resources: [.copy("Snapshots")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
