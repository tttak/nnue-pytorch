#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_training_data_formats.h"
#include "../YaneuraOu/source/learn/learn.h"

#include <optional>
#include <fstream>
#include <string>
#include <memory>

#include <ppl.h>

namespace training_data {

    using namespace binpack;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size()) return false;

        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    static std::string filename_with_extension(const std::string& filename, const std::string& ext)
    {
        if (ends_with(filename, ext))
        {
            return filename;
        }
        else
        {
            return filename + "." + ext;
        }
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;
        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                {
                    break;
                }
                vec.emplace_back(*v);
            }
        }

        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}
    };

    struct BinSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputStream(std::string filename1, std::string filename2, std::string filename3, float train1_rate, float train2_rate, float skiprate, float mirror, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream1(filename1, openmode),
            m_stream2(filename2, openmode),
            m_stream3(filename3, openmode),
            m_filename1(filename1),
            m_filename2(filename2),
            m_filename3(filename3),
            m_train1_rate(train1_rate),
            m_train2_rate(train2_rate),
            m_skiprate(skiprate),
            m_mirror(mirror),
            m_eof1(!m_stream1),
            m_eof2(!m_stream2),
            m_eof3(!m_stream3),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            Learner::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for(;;)
            {
                if(m_stream1.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
                {
                    auto entry = packedSfenValueToTrainingDataEntry(e);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return std::nullopt;

                        std::cout << "next()!!" << "m_stream1 = std::fstream(m_filename1, openmode);" << m_filename1 << std::endl;

                        m_stream1 = std::fstream(m_filename1, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream1)
                            return std::nullopt;

                        continue;
                    }

                    m_eof1 = true;
                    return std::nullopt;
                }
            }
        }


        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
          vec.resize(n);
          std::size_t remain1 = n;

          while (remain1 > 0) {
            std::size_t remain2 = remain1 + 10;
            std::vector<TrainingDataEntry> work_vec(remain2);

            std::size_t n1 = remain2 * m_train1_rate;
            std::size_t n2 = remain2 * m_train2_rate;
            std::size_t n3 = remain2 - n1 - n2;

            std::size_t m1 = n1 * m_skiprate;
            std::size_t m2 = n2 * m_skiprate;
            std::size_t m3 = n3 * m_skiprate;

            std::vector<Learner::PackedSfenValue> packedSfenValues1(m1);
            std::vector<Learner::PackedSfenValue> packedSfenValues2(m2);
            std::vector<Learner::PackedSfenValue> packedSfenValues3(m3);

            for (int i=0;i<2;i++) {
                if (m_stream1.read(reinterpret_cast<char*>(&packedSfenValues1[0]), sizeof(Learner::PackedSfenValue) * m1)) {
                    break;
                }
                else {
                    std::cout << "fill()!!" << "m_stream1 = std::fstream(m_filename1, openmode);" << m_filename1 << std::endl;
                    m_stream1 = std::fstream(m_filename1, openmode);
                }
            }

            for (int i=0;i<2;i++) {
                if (m_stream2.read(reinterpret_cast<char*>(&packedSfenValues2[0]), sizeof(Learner::PackedSfenValue) * m2)) {
                    break;
                }
                else {
                    std::cout << "fill()!!" << "m_stream2 = std::fstream(m_filename2, openmode);" << m_filename2 << std::endl;
                    m_stream2 = std::fstream(m_filename2, openmode);
                }
            }

            for (int i=0;i<2;i++) {
                if (m_stream3.read(reinterpret_cast<char*>(&packedSfenValues3[0]), sizeof(Learner::PackedSfenValue) * m3)) {
                    break;
                }
                else {
                    std::cout << "fill()!!" << "m_stream3 = std::fstream(m_filename3, openmode);" << m_filename3 << std::endl;
                    m_stream3 = std::fstream(m_filename3, openmode);
                }
            }

            std::shuffle(packedSfenValues1.begin(), packedSfenValues1.end(), m_rand);
            std::shuffle(packedSfenValues2.begin(), packedSfenValues2.end(), m_rand);
            std::shuffle(packedSfenValues3.begin(), packedSfenValues3.end(), m_rand);

            PRNG prng;
            float mirror = m_mirror;

            concurrency::parallel_for(size_t(0), n1, [&work_vec, &packedSfenValues1, &prng, &mirror](size_t i)
            {
                const bool mir = prng.rand(1000000) < mirror * 1000000.0f;
                work_vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues1[i], mir);
            });
            concurrency::parallel_for(size_t(0), n2, [&work_vec, &packedSfenValues2, &n1, &prng, &mirror](size_t i)
            {
                const bool mir = prng.rand(1000000) < mirror * 1000000.0f;
                work_vec[n1+i] = packedSfenValueToTrainingDataEntry(packedSfenValues2[i], mir);
            });
            concurrency::parallel_for(size_t(0), n3, [&work_vec, &packedSfenValues3, &n1, &n2, &prng, &mirror](size_t i)
            {
                const bool mir = prng.rand(1000000) < mirror * 1000000.0f;
                work_vec[n1+n2+i] = packedSfenValueToTrainingDataEntry(packedSfenValues3[i], mir);
            });

            int j = n - remain1;
            for (int i = 0; i < remain2; i++) {
                if (!work_vec[i].skip) {
                    vec[j] = work_vec[i];
                    j++;

                    if (j >= n) {
                        break;
                    }
                }
            }

            remain1 = n - j;
          }

        }

        bool eof() const override
        {
            return m_eof1;
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream1;
        std::fstream m_stream2;
        std::fstream m_stream3;
        std::string m_filename1;
        std::string m_filename2;
        std::string m_filename3;
        float m_train1_rate;
        float m_train2_rate;
        float m_skiprate;
        float m_mirror;
        bool m_eof1;
        bool m_eof2;
        bool m_eof3;
        bool m_cyclic;
        std::mt19937_64 m_rand;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename1, const std::string& filename2, const std::string& filename3, float train1_rate, float train2_rate, float skiprate, float mirror, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename1, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, cyclic, std::move(skipPredicate));

        return nullptr;
    }

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::string& filename1, const std::string& filename2, const std::string& filename3, float train1_rate, float train2_rate, float skiprate, float mirror, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        // TODO (low priority): optimize and parallelize .bin reading.
        if (has_extension(filename1, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, cyclic, std::move(skipPredicate));

        return nullptr;
    }
}

#endif
