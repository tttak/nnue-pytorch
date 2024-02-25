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
            for(;;)
            {
                if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
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
            std::vector<Learner::PackedSfenValue> packedSfenValues(n);
            bool reopenedFileOnce = false;
            for (;;)
            {
                if (m_stream.read(reinterpret_cast<char*>(&packedSfenValues[0]), sizeof(Learner::PackedSfenValue) * n))
                {
                    vec.resize(n);
                    concurrency::parallel_for(size_t(0), n, [&vec, &packedSfenValues](size_t i)
                        {
                            vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                        });
                    return;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return;

                        continue;
                    }

                    m_eof = true;
                    return;
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

        BinSfenInputParallelStream(int concurrency, std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate, int batch_size) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_concurrency(concurrency),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate)),
            m_batch_size(batch_size)
        {
            m_stop_flag.store(false);

            auto worker = [this]()
            {
                while (!m_stop_flag.load())
                {
                    auto packedSfenValues = std::make_shared<std::vector<Learner::PackedSfenValue>>(m_batch_size);

                    {
                        std::unique_lock lock(m_stream_mutex);

                        bool reopenedFileOnce = false;
                        for (;;)
                        {
                            if (m_stream.read(reinterpret_cast<char*>(&(*packedSfenValues)[0]), sizeof(Learner::PackedSfenValue) * m_batch_size))
                            {
                                break;
                            }
                            else
                            {
                                if (m_cyclic)
                                {
                                    if (reopenedFileOnce)
                                        return;

                                    m_stream = std::fstream(m_filename, openmode);
                                    reopenedFileOnce = true;
                                    if (!m_stream)
                                        return;

                                    continue;
                                }

                                m_eof = true;
                                return;
                            }
                        }
                    }

                    {
                        std::unique_lock lock(m_batch_mutex);
                        m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                        m_batches.emplace_back(packedSfenValues);

                        lock.unlock();
                        m_batches_any.notify_one();
                    }

                }
                m_num_workers.fetch_sub(1);
                m_batches_any.notify_one();
            };
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
            std::vector<Learner::PackedSfenValue> packedSfenValues(n);
            bool reopenedFileOnce = false;
            for (;;)
            {
                if (m_stream.read(reinterpret_cast<char*>(&packedSfenValues[0]), sizeof(Learner::PackedSfenValue) * n))
                {
                    vec.resize(n);
                    concurrency::parallel_for(size_t(0), n, [&vec, &packedSfenValues](size_t i)
                        {
                            vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                        });
                    return;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return;

                        continue;
                    }

                    m_eof = true;
                    return;
                }
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputParallelStream() override
        {
            m_stop_flag.store(true);
            m_batches_not_full.notify_all();

            for (auto& worker : m_workers)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }
        }

    private:
        std::fstream m_stream;
        std::string m_filename;
        int m_concurrency;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        int m_batch_size;
        std::deque<std::shared_ptr<std::vector<Learner::PackedSfenValue>>> m_batches;
        std::mutex m_batch_mutex;
        std::mutex m_stream_mutex;
        std::condition_variable m_batches_not_full;
        std::condition_variable m_batches_any;
        std::atomic_bool m_stop_flag;
        std::atomic_int m_num_workers;
        std::vector<std::thread> m_workers;

        std::shared_ptr<std::vector<Learner::PackedSfenValue>> next_packed_sfen_values()
        {
            std::unique_lock lock(m_batch_mutex);
            m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

            if (!m_batches.empty())
            {
                auto batch = m_batches.front();
                m_batches.pop_front();

                lock.unlock();
                m_batches_not_full.notify_one();

                return batch;
            }
            return nullptr;
        }

    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate, int batch_size)
    {
        // TODO (low priority): optimize and parallelize .bin reading.
        if (has_extension(filename, BinSfenInputParallelStream::extension))
            //return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));
            return std::make_unique<BinSfenInputParallelStream>(concurrency, filename, cyclic, std::move(skipPredicate), batch_size);

        return nullptr;
    }
}

#endif
