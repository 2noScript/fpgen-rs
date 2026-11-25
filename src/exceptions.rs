use std::io;
use thiserror::Error;

/// Errors returned by `fpgen-rs`, preserving the Python/TypeScript exception
/// hierarchy and variant names (`InvalidConstraints`, `RestrictiveConstraints`,
/// `InvalidNode`, `NodePathError`, `ModelNotInstalled`, `ModelVerificationError`, etc.).
#[derive(Debug, Error)]
pub enum FpgenError {
    #[error("{0}")]
    ValueError(String),

    #[error("{0}")]
    NetworkError(String),

    #[error("{0}")]
    InvalidConstraints(String),

    #[error("{0}")]
    RestrictiveConstraints(String),

    #[error("{0}")]
    InvalidNode(String),

    #[error("{0}")]
    NodePathError(String),

    #[error("{0}")]
    ModelNotInstalled(String),

    #[error("{0}")]
    ModelVerificationError(String),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

impl FpgenError {
    /// Returns the canonical Python/TypeScript exception class name for this error.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ValueError(_) => "ValueError",
            Self::NetworkError(_) => "NetworkError",
            Self::InvalidConstraints(_) => "InvalidConstraints",
            Self::RestrictiveConstraints(_) => "RestrictiveConstraints",
            Self::InvalidNode(_) => "InvalidNode",
            Self::NodePathError(_) => "NodePathError",
            Self::ModelNotInstalled(_) => "ModelNotInstalled",
            Self::ModelVerificationError(_) => "ModelVerificationError",
            Self::Io(_) => "IoError",
        }
    }
}

pub type Result<T> = std::result::Result<T, FpgenError>;
