//! Neural-network operations used by Sift's inference models.
//!
//! This crate intentionally exposes only the operations used by those models.

pub mod embedding;
pub mod linear;
pub mod ops;
pub mod rotary_emb;
pub mod var_builder;

pub use crate::Module;
pub use embedding::{embedding, Embedding};
pub use linear::{linear_no_bias, Linear};
pub use var_builder::VarBuilder;
