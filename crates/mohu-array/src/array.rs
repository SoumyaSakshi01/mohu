use std::marker::PhantomData;

use mohu_buffer::Buffer;
use mohu_dtype::Scalar;
use mohu_dtype::DType;
use mohu_error::MohuResult;

pub struct NdArray<T: Scalar> {
    buffer: Buffer,
    marker: PhantomData<T>,
}

impl<T: Scalar> NdArray<T> {
    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        self.buffer.shape()
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn zero(shape: &[usize]) -> MohuResult<Self> {
        let buffer = Buffer::zeros(T::DTYPE, shape)?;
        Ok(Self { buffer, marker: PhantomData })
    }

    pub fn ones(shape: &[usize]) -> MohuResult<Self> {
        let buffer = Buffer::ones(T::DTYPE, shape)?;
        Ok(Self { buffer, marker: PhantomData })
    }

    pub fn from_sliced(data: &[T]) -> MohuResult<Self> {
        let buffer = Buffer::from_slice(data)?;
        Ok(Self { buffer, marker: PhantomData })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_shape_and_len() {
        let a = NdArray::<f64>::zero(&[3, 4]).unwrap();

        assert_eq!(a.shape(), &[3, 4]);
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.len(), 12);
        assert!(a.is_empty() == false);
        assert_eq!(a.dtype(), mohu_dtype::DType::F64);
    }

    #[test]
    fn test_from_slice_len() {
        let a = NdArray::<f32>::from_sliced(&[1.0, 2.0, 3.0]).unwrap();

        assert_eq!(a.len(), 3);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.dtype(), DType::F32);
        assert!(!a.is_empty());
    }

    #[test]
    fn test_dtype_f32() {
        let a = NdArray::<f32>::zero(&[2, 2]).unwrap();
        assert_eq!(a.dtype(), DType::F32);
    }

    #[test]
    fn test_dtype_f64() {
        let a = NdArray::<f64>::zero(&[2, 2]).unwrap();
        assert_eq!(a.dtype(), DType::F64);
    }
}