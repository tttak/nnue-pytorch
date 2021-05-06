#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_training_data_formats.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

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

        BinSfenInputStream(std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            Learner::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for (;;)
            {
                if (m_stream.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
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

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return std::nullopt;

                        continue;
                    }

                    m_eof = true;
                    return std::nullopt;
                }
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream;
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
    };

    struct BinSfenInputParallelStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputParallelStream(int concurrency, std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_concurrency(concurrency),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate)),
            m_filesize(std::filesystem::file_size(m_filename))
        {
        }

        std::optional<TrainingDataEntry> next() override
        {
            Learner::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for (;;)
            {
                if (m_stream.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
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

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return std::nullopt;

                        continue;
                    }

                    m_eof = true;
                    return std::nullopt;
                }
            }
        }

        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            std::vector<Learner::PackedSfenValue> e(n);
            int entry_index = 0;
            while (entry_index < n) {
                auto expected_read_size = sizeof(Learner::PackedSfenValue) * (n - entry_index);
                auto actual_read_size = std::min(m_filesize - m_stream.tellg(), expected_read_size);
                m_stream.read(reinterpret_cast<char*>(&e[entry_index]), actual_read_size);
                if (expected_read_size == actual_read_size) {
                    // 指定された分だけ読み終えた。
                    break;
                }

                if (!m_cyclic) {
                    // 一度しか読まないので、eofフラグを立てて、returnする。
                    m_eof = true;
                    return;
                }

                m_stream = std::fstream(m_filename, openmode);

                if (!m_stream) {
                    // ファイルの再読み込みに失敗したので、eofフラグを立てて、returnする。
                    m_eof = true;
                    return;
                }

                entry_index += actual_read_size / sizeof(Learner::PackedSfenValue);
            }

            vec.resize(n);

            concurrency::parallel_for(0, static_cast<int>(n), [&vec, &e](int entry_index) {
                // TODO(hnoda): m_skipPredicateとm_skipPredicate(entry)を考慮する。
                vec[entry_index] = packedSfenValueToTrainingDataEntry(e[entry_index]);
                });
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputParallelStream() override {}

    private:
        int m_concurrency;
        std::fstream m_stream;
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        uintmax_t m_filesize;

    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputParallelStream::extension))
            return std::make_unique<BinSfenInputParallelStream>(concurrency, filename, cyclic, std::move(skipPredicate));
        //if (has_extension(filename, BinSfenInputStream::extension))
        //    return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }
}

#endif
