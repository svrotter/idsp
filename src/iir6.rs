use miniconf::MiniconfAtomic;
use serde::{Deserialize, Serialize};

use super::macc;
use core::iter::Sum;
use num_traits::{clamp, Float};

/// IIR state and coefficients type.
///
/// To represent the IIR state (input and output memory) during the filter update
/// this contains the seven inputs (x0...x6) and the six outputs (y1...y6)
/// concatenated. Lower indices correspond to more recent samples.
/// To represent the IIR coefficients, this contains the feed-forward
/// coefficients (b0...b6) followed by the negated feed-back coefficients
/// (-a1...-a6). These are normalized such that a0 = 1.
pub type Vec13<T> = [T; 13];

/// IIR configuration.
///
/// Contains the coeeficients `ba`, the output offset `y_offset`, and the
/// output limits `y_min` and `y_max`.
///
/// This implementation achieves several important properties:
///
/// * Its transfer function is universal in the sense that any 6th order
///   transfer function can be implemented without code
///   changes preserving all features.
/// * It inherits a universal implementation of "integrator anti-windup", also
///   and especially in the presence of set-point changes and in the presence
///   of proportional or derivative gain without any back-off that would reduce
///   steady-state output range.
/// * An offset at the input of an IIR filter (a.k.a. "set-point") is
///   equivalent to an offset at the output. They are related by the
///   overall (DC feed-forward) gain of the filter.
/// * It stores only previous outputs and inputs. These have direct and
///   invariant interpretation (independent of gains and offsets).
///   Therefore it can trivially implement bump-less transfer.
///
/// # Miniconf
///
/// `{"y_offset": y_offset, "y_min": y_min, "y_max": y_max, "ba": [b0...b6, -a1...-a6]}`
///
/// * `y0` is the output offset code
/// * `ym` is the lower saturation limit
/// * `yM` is the upper saturation limit
///
/// IIR filter tap gains (`ba`) are an array `[b0...b6, -a1...-a6]` such that the
/// new output is computed as `y0 = bi*xi - aj*yj, i=0...6, j=1...6.
#[derive(Copy, Clone, Debug, Default, Deserialize, Serialize, MiniconfAtomic)]
pub struct IIR6<T> {
    pub ba: Vec13<T>,
    pub y_offset: T,
    pub y_min: T,
    pub y_max: T,
}

impl<T: Float + Default + Sum<T>> IIR6<T> {
    pub fn new(y_min: T, y_max: T) -> Self {
        Self {
            ba: [T::default(); 13],
            y_offset: T::default(),
            y_min,
            y_max,
        }
    }

    /// Feed a new input value into the filter, update the filter state, and
    /// return the new output. Only the state `xy` is modified.
    ///
    /// # Arguments
    /// * `xy` - Current filter state.
    /// * `x0` - New input.
    pub fn update(&self, xy: &mut Vec13<T>, x0: T, hold: bool) -> T {
        let n = self.ba.len();
        debug_assert!(xy.len() == n);
        // `xy` contains       [x0...x5, y0... y6]
        // Increment time      [x1...x6, y1... y7]
        // Shift               [x1, x1...x6, y1... y6]
        // This unrolls better than xy.rotate_right(1)
        xy.copy_within(0..n - 1, 1);
        // Store x0            [x0...x6, y1... y6]
        xy[0] = x0;
        // Compute y0 by multiply-accumulate
        let y0 = if hold {
            xy[n / 2 + 1]
        } else {
            macc(self.y_offset, xy, &self.ba)
        };
        // Limit y0
        let y0 = clamp(y0, self.y_min, self.y_max);
        // Store y0            [x0...x5, y0... y6]
        xy[n / 2] = y0;
        return y0;
    }
}
