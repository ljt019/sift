pub trait UnaryOpT {
    const NAME: &'static str;
    #[cfg(feature = "cuda")]
    const KERNEL: &'static str;
    #[cfg(feature = "cuda")]
    const V: Self;
    fn f32(value: f32) -> f32;

    #[inline(always)]
    fn f32_vec(input: &[f32], output: &mut [f32]) {
        input
            .iter()
            .zip(output)
            .for_each(|(&value, output)| *output = Self::f32(value));
    }
}

pub trait BinaryOpT {
    const NAME: &'static str;
    #[cfg(feature = "cuda")]
    const KERNEL: &'static str;
    #[cfg(feature = "cuda")]
    const V: Self;
    fn f32(lhs: f32, rhs: f32) -> f32;

    #[inline(always)]
    fn f32_vec(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
        lhs.iter()
            .zip(rhs)
            .zip(output)
            .for_each(|((&lhs, &rhs), output)| *output = Self::f32(lhs, rhs));
    }

    #[inline(always)]
    fn f32_scalar_vec(scalar: f32, input: &[f32], output: &mut [f32]) {
        input
            .iter()
            .zip(output)
            .for_each(|(&value, output)| *output = Self::f32(value, scalar));
    }
}

macro_rules! binary_op {
    ($name:ident, $kernel:literal, $op:tt) => {
        pub struct $name;

        impl BinaryOpT for $name {
            const NAME: &'static str = $kernel;
            #[cfg(feature = "cuda")]
            const KERNEL: &'static str = concat!("b", $kernel);
            #[cfg(feature = "cuda")]
            const V: Self = Self;

            #[inline(always)]
            fn f32(lhs: f32, rhs: f32) -> f32 {
                lhs $op rhs
            }
        }
    };
}

binary_op!(Add, "add", +);
binary_op!(Mul, "mul", *);
binary_op!(Div, "div", /);

macro_rules! unary_op {
    ($name:ident, $kernel:literal, $body:expr) => {
        pub struct $name;

        impl UnaryOpT for $name {
            const NAME: &'static str = $kernel;
            #[cfg(feature = "cuda")]
            const KERNEL: &'static str = concat!("u", $kernel);
            #[cfg(feature = "cuda")]
            const V: Self = Self;

            #[inline(always)]
            fn f32(value: f32) -> f32 {
                $body(value)
            }
        }
    };
}

unary_op!(Sin, "sin", f32::sin);
unary_op!(Cos, "cos", f32::cos);
unary_op!(Sqr, "sqr", |value| value * value);
unary_op!(Sqrt, "sqrt", f32::sqrt);

pub struct Gelu;

impl UnaryOpT for Gelu {
    const NAME: &'static str = "gelu";
    #[cfg(feature = "cuda")]
    const KERNEL: &'static str = "ugelu";
    #[cfg(feature = "cuda")]
    const V: Self = Self;

    #[inline(always)]
    fn f32(value: f32) -> f32 {
        const SQRT_TWO_OVER_PI: f32 = 0.797_884_6;
        0.5 * value * (1.0 + (SQRT_TWO_OVER_PI * value * (1.0 + 0.044_715 * value * value)).tanh())
    }
}

pub struct GeluErf;

impl UnaryOpT for GeluErf {
    const NAME: &'static str = "gelu_erf";
    #[cfg(feature = "cuda")]
    const KERNEL: &'static str = "ugelu_erf";
    #[cfg(feature = "cuda")]
    const V: Self = Self;

    #[inline(always)]
    fn f32(value: f32) -> f32 {
        0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
    }
}
