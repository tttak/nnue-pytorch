/*

Copyright 2020 Tomasz Sobczyk

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files (the "Software"),
to deal in the Software without restriction,
including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#pragma once

#include <cstdio>
#include <cassert>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>
#include <iostream>
#include <set>
#include <cstdio>
#include <cassert>
#include <array>
#include <limits>
#include <climits>
#include <optional>
#include <thread>
#include <mutex>
#include <random>

#include "rng.h"
#include "YaneuraOu/learn/learn.h"
#include "YaneuraOu/movepick.h"
#include "YaneuraOu/position.h"
#include "YaneuraOu/thread.h"

#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && !defined(__clang__)
#include <intrin.h>
#endif

namespace binpack
{
    struct TrainingDataEntry
    {
        std::shared_ptr<Position> pos;
        Move move;
        std::int16_t score;
        std::uint16_t ply;
        std::int16_t result;

        [[nodiscard]] bool isValid() const
        {
            return pos->pseudo_legal(move) && pos->legal(move);
        }

        [[nodiscard]] bool isCapturingMove() const
        {
            return pos->piece_on(to_sq(move)) != Piece::NO_PIECE;
        }

        [[nodiscard]] bool isInCheck() const
        {
            return pos->in_check();
        }
    };

    [[nodiscard]] inline TrainingDataEntry packedSfenValueToTrainingDataEntry(const Learner::PackedSfenValue& psv)
    {
        TrainingDataEntry ret;

        ret.pos = std::make_shared<Position>();
        StateInfo state_info = {};
        ret.pos->set_from_packed_sfen(psv.sfen, &state_info, Threads[0]);
        ret.move = ret.pos->to_move(psv.move);
        ret.score = psv.score;
        ret.ply = psv.gamePly;
        ret.result = psv.game_result;

        return ret;
    }
}
