//
//  Token.swift
//  Kototama
//
//  The three numeric types that flow through every llama.cpp call.
//

/// A token ID: an index into the model's vocabulary.
///
/// The Rust crate aliases this to `i32`; we use a dedicated struct instead so
/// that function signatures read `Token` rather than a bare `Int32`, and so a
/// token can never be silently used as a position or a sequence ID.
public struct Token: Hashable, Sendable, Codable, Comparable, ExpressibleByIntegerLiteral, CustomStringConvertible {
    /// The raw token ID, as llama.cpp uses it (`llama_token`).
    public let rawValue: Int32

    public init(_ rawValue: Int32) {
        self.rawValue = rawValue
    }

    public init(integerLiteral value: Int32) {
        self.rawValue = value
    }

    public static func < (lhs: Token, rhs: Token) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public var description: String { "Token(\(rawValue))" }

    /// The "no token" sentinel llama.cpp uses (`LLAMA_TOKEN_NULL`, -1).
    public static let none = Token(-1)
}

/// A position index within a sequence (0-based).
public struct Position: Hashable, Sendable, Codable, Comparable, ExpressibleByIntegerLiteral, CustomStringConvertible {
    public let rawValue: Int32

    public init(_ rawValue: Int32) {
        self.rawValue = rawValue
    }

    public init(integerLiteral value: Int32) {
        self.rawValue = value
    }

    public static func < (lhs: Position, rhs: Position) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public var description: String { "Position(\(rawValue))" }

    /// The "no position" sentinel llama.cpp reports for empty sequences (-1).
    public static let none = Position(-1)
}

/// A sequence ID: one of the independent generation slots inside a `Context`.
public struct SequenceID: Hashable, Sendable, Codable, Comparable, ExpressibleByIntegerLiteral, CustomStringConvertible {
    public let rawValue: Int32

    public init(_ rawValue: Int32) {
        self.rawValue = rawValue
    }

    public init(integerLiteral value: Int32) {
        self.rawValue = value
    }

    public static func < (lhs: SequenceID, rhs: SequenceID) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public var description: String { "SequenceID(\(rawValue))" }
}
