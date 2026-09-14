// Diagnostic architecture used by the bundled tanuki-dr5 network.
#ifndef NNUE_HALFKP_512X2_8_96_H_INCLUDED
#define NNUE_HALFKP_512X2_8_96_H_INCLUDED

#include "../features/feature_set.h"
#include "../features/half_kp.h"
#include "../layers/input_slice.h"
#include "../layers/affine_transform.h"
#include "../layers/clipped_relu.h"

namespace Eval::NNUE {

using RawFeatures = Features::FeatureSet<
    Features::HalfKP<Features::Side::kFriend>>;
constexpr IndexType kTransformedFeatureDimensions = 512;

namespace Layers {
using InputLayer = InputSlice<kTransformedFeatureDimensions * 2>;
using HiddenLayer1 = ClippedReLU<AffineTransform<InputLayer, 8>>;
using HiddenLayer2 = ClippedReLU<AffineTransform<HiddenLayer1, 96>>;
using OutputLayer = AffineTransform<HiddenLayer2, 1>;
}

using Network = Layers::OutputLayer;

}

#endif
