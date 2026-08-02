use crate::Tensor;

impl std::fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("shape", self.shape())
            .field("dtype", &self.dtype())
            .field("device", self.device())
            .finish()
    }
}
