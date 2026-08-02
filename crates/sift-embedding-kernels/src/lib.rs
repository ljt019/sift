#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

pub const CUDA_UTILS_HEADER: &str = include_str!("cuda_utils.cuh");
pub const BINARY_OP_MACROS_HEADER: &str = include_str!("binary_op_macros.cuh");

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Indexing,
    Matmul,
    Reduce,
    Unary,
}

pub const ALL_IDS: [Id; 7] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Indexing,
    Id::Matmul,
    Id::Reduce,
    Id::Unary,
];

pub struct Module {
    index: usize,
    name: &'static str,
    source: &'static str,
    #[cfg(feature = "cuda")]
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn source(&self) -> &'static str {
        self.source
    }

    #[cfg(feature = "cuda")]
    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident, $source:literal) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            name: concat!($source, ".cu"),
            source: include_str!(concat!($source, ".cu")),
            #[cfg(feature = "cuda")]
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine, "affine");
mdl!(BINARY, Binary, "binary");
mdl!(CAST, Cast, "cast");
mdl!(INDEXING, Indexing, "indexing");
mdl!(MATMUL, Matmul, "matmul");
mdl!(REDUCE, Reduce, "reduce");
mdl!(UNARY, Unary, "unary");
