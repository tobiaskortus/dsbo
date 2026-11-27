#pragma once

#include "types.hpp"
#include "internal/core.hpp"
#include "internal/datasets.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dsbo::io {

// Reader handles must not cross fork/process boundaries. Calls on one reader,
// including close(), are serialized; unrelated readers use HDF5's own locking.
class Reader {
    struct MappedFile {
        int fd = -1;
        void* data = nullptr;
        std::size_t size = 0;
        MappedFile() = default;
        MappedFile(const MappedFile&) = delete;
        MappedFile& operator=(const MappedFile&) = delete;
        ~MappedFile() { reset(); }
        bool valid() const noexcept { return data != nullptr; }
        bool map() noexcept {
            if (valid()) return true;
            if (fd < 0 || size == 0) return false;
            void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
            if (mapped == MAP_FAILED) return false;
            ::madvise(mapped, size, MADV_NORMAL);
            data = mapped;
            return true;
        }
        void advise(int advice) const noexcept {
            if (data) ::madvise(data, size, advice);
        }
        void reset() noexcept {
            if (data) {
                ::munmap(data, size);
                data = nullptr;
            }
            if (fd >= 0) {
                ::close(fd);
                fd = -1;
            }
            size = 0;
        }
    };

    // One physical source file behind a DSBO virtual concatenation. The
    // descriptor is duplicated at open time so mappings survive path changes.
    struct SourceFile {
        internal::Handle file;
        MappedFile mapping;
        bool failed = false;
        SourceFile() = default;
        SourceFile(const SourceFile&) = delete;
        SourceFile& operator=(const SourceFile&) = delete;
    };

    // A contiguous row range of the virtual dataset backed by one source file.
    struct Partition {
        std::shared_ptr<SourceFile> source;
        std::size_t virtual_start = 0;
        std::size_t rows = 0;
        std::size_t data_offset = 0;
        std::size_t row_bytes = 0;
    };

    struct CachedDataset {
        internal::Handle dataset;
        internal::Handle space;
        Shape shape;
        DType type;
        std::size_t row_bytes = 0;
        bool mapped = false;
        std::size_t offset = 0;
        const unsigned char* mapped_base = nullptr;
        bool mapped_base_checked = false;
        bool vds = false;
        bool vds_checked = false;
        std::vector<Partition> partitions;
        std::vector<std::size_t> cumulative;
    };

    internal::Handle file_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, CachedDataset> cache_;
    mutable MappedFile mapping_;
    mutable bool mapping_failed_ = false;
    bool sec2_driver_ = false;
    std::vector<std::shared_ptr<SourceFile>> sources_;
    bool sources_failed_ = false;

    void require_open() const { if (!file_) throw std::runtime_error("HDF5 reader has been closed."); }

    // Duplicate the descriptor behind an open HDF5 file and record its size.
    // Returns whether the file uses the sec2 driver, the only driver whose
    // logical file offsets match a plain mmap of the file.
    static bool duplicate_sec2_descriptor(hid_t file, int& fd, std::size_t& size) {
        bool sec2 = false;
        H5E_BEGIN_TRY {
            const hid_t access = H5Fget_access_plist(file);
            if (access >= 0) {
                sec2 = H5Pget_driver(access) == H5FD_SEC2;
                if (sec2) {
                    void* handle = nullptr;
                    if (H5Fget_vfd_handle(file, H5P_DEFAULT, &handle) >= 0 && handle)
                        fd = ::dup(*static_cast<const int*>(handle));
                }
                H5Pclose(access);
            }
        } H5E_END_TRY;
        if (fd < 0) return sec2;
        struct stat info{};
        if (::fstat(fd, &info) == 0 && info.st_size > 0) {
            size = static_cast<std::size_t>(info.st_size);
        } else {
            ::close(fd);
            fd = -1;
        }
        return sec2;
    }

    bool ensure_mapping_locked() const {
        if (mapping_failed_) return false;
        if (mapping_.map()) return true;
        if (mapping_.fd >= 0 && mapping_.size != 0) mapping_failed_ = true;
        return false;
    }

    static std::size_t row_bytes_for(const Shape& dimensions, DType type) {
        const auto trailing = elements(Shape(dimensions.begin() + 1, dimensions.end()));
        if (!type.size || trailing > std::numeric_limits<std::size_t>::max() / type.size)
            throw std::overflow_error("Array row size exceeds addressable memory.");
        return trailing * type.size;
    }

    static void normalize_indices(const Shape& dimensions, std::vector<std::int64_t>& indices) {
        if (dimensions.empty()) throw std::invalid_argument("HDF5 dataset has no dimensions.");
        const auto rows = dimensions[0];
        for (auto& index : indices) {
            if (index < 0) index += static_cast<std::int64_t>(rows);
            if (index < 0 || static_cast<hsize_t>(index) >= rows)
                throw std::out_of_range("HDF5 dataset index is out of range.");
        }
    }

    static void prefetch_read(const void* address) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(address, 0, 3);
#else
        (void)address;
#endif
    }

    static Shape filled_shape(std::size_t rank, hsize_t value) {
        return Shape(std::vector<hsize_t>(rank, value));
    }

    static std::optional<std::size_t> contiguous_mapping(hid_t dataset, hid_t native_type, bool numeric_only) {
        internal::Handle properties(H5Dget_create_plist(dataset), H5Pclose);
        if (H5Pget_layout(properties) != H5D_CONTIGUOUS || H5Pget_external_count(properties) != 0)
            return std::nullopt;
        H5D_space_status_t status;
        if (H5Dget_space_status(dataset, &status) < 0 || status != H5D_SPACE_STATUS_ALLOCATED)
            return std::nullopt;
        const auto offset = H5Dget_offset(dataset);
        if (offset == HADDR_UNDEF) return std::nullopt;
        internal::Handle file_type(H5Dget_type(dataset), H5Tclose);
        if (numeric_only) {
            const auto kind = H5Tget_class(file_type);
            if (kind != H5T_INTEGER && kind != H5T_FLOAT) return std::nullopt;
        }
        if (H5Tequal(file_type, native_type) <= 0) return std::nullopt;
        return static_cast<std::size_t>(offset);
    }

    // Validate DSBO's concatenation metadata against the physical sources and
    // build the global-row -> (source, local row) table. Any mismatch leaves
    // the dataset unmapped so reads fall back to HDF5 selection I/O.
    void prepare_vds_locked(CachedDataset& entry, const std::string& name) const {
        internal::Handle properties(H5Dget_create_plist(entry.dataset), H5Pclose);
        if (H5Pget_layout(properties) != H5D_VIRTUAL) return;
        const auto sizes = internal::json_attribute<std::vector<hsize_t>>(entry.dataset, partitions_attribute);
        if (!sizes || sizes->size() != sources_.size()) return;
        std::uint64_t total = 0;
        for (const auto size : *sizes) {
            if (size > std::numeric_limits<std::uint64_t>::max() - total) return;
            total += size;
        }
        if (entry.shape.empty() || total != entry.shape[0]) return;
        const auto row_bytes = entry.row_bytes;
        internal::Handle native_type = internal::datatype(entry.type);

        std::vector<Partition> partitions;
        std::vector<std::size_t> cumulative{0};
        partitions.reserve(sources_.size());
        std::size_t row_start = 0;
        for (std::size_t index = 0; index < sources_.size(); ++index) {
            const auto& source = sources_[index];
            if (source->failed || source->mapping.fd < 0 || source->mapping.size == 0) return;
            const auto rows = static_cast<std::uint64_t>((*sizes)[index]);
            if (rows == 0) continue;
            internal::Handle dataset;
            try { dataset = internal::dataset(source->file, name); }
            catch (const std::bad_alloc&) { throw; }
            catch (...) { return; }
            const auto source_shape = internal::dataset_shape(dataset);
            if (source_shape.empty()) {
                if (rows != 1) return;
            } else {
                if (source_shape.size() != entry.shape.size() || source_shape[0] != rows) return;
                if (!std::equal(entry.shape.begin() + 1, entry.shape.end(), source_shape.begin() + 1)) return;
            }
            if (internal::dataset_dtype(dataset) != entry.type) return;
            const auto offset = contiguous_mapping(dataset, native_type, false);
            if (!offset) return;
            const auto byte_offset = static_cast<std::uint64_t>(*offset);
            if (row_bytes && rows > std::numeric_limits<std::uint64_t>::max() / row_bytes) return;
            const auto payload = rows * static_cast<std::uint64_t>(row_bytes);
            if (byte_offset > std::numeric_limits<std::uint64_t>::max() - payload) return;
            if (byte_offset + payload > static_cast<std::uint64_t>(source->mapping.size)) return;
            partitions.push_back({source, row_start, static_cast<std::size_t>(rows),
                                  static_cast<std::size_t>(byte_offset), row_bytes});
            row_start += static_cast<std::size_t>(rows);
            cumulative.push_back(row_start);
        }
        if (row_start != entry.shape[0]) return;
        entry.vds = true;
        entry.partitions = std::move(partitions);
        entry.cumulative = std::move(cumulative);
    }

    std::size_t partition_index(const CachedDataset& cached, std::size_t row) const {
        const auto it = std::upper_bound(cached.cumulative.begin(), cached.cumulative.end(), row);
        return static_cast<std::size_t>(it - cached.cumulative.begin() - 1);
    }

    bool copy_vds_range_locked(const CachedDataset& cached, std::size_t first, std::size_t count,
                               unsigned char* output) const {
        auto index = partition_index(cached, first);
        std::size_t remaining = count;
        while (remaining) {
            const auto& partition = cached.partitions[index];
            if (partition.source->failed) return false;
            if (!partition.source->mapping.map()) {
                partition.source->failed = true;
                return false;
            }
            const auto local = first - partition.virtual_start;
            const auto take = std::min(remaining, partition.rows - local);
            const auto* base = static_cast<const unsigned char*>(partition.source->mapping.data) + partition.data_offset;
            std::memcpy(output, base + local * partition.row_bytes, take * partition.row_bytes);
            output += take * partition.row_bytes;
            first += take;
            remaining -= take;
            ++index;
        }
        return true;
    }

    void copy_scalar_rows_locked(const CachedDataset& cached, std::size_t count,
                                 unsigned char* output) const {
        auto scalar = internal::read_dataset(cached.dataset);
        const auto* source = scalar.data();
        const auto scalar_bytes = scalar.nbytes();
        for (std::size_t i = 0; i < count; ++i)
            std::memcpy(output + i * scalar_bytes, source, scalar_bytes);
    }

    bool copy_vds_range_if_available_locked(const CachedDataset& cached, std::size_t first,
                                            std::size_t count, unsigned char* output) const {
        if (!cached.vds) return false;
        return copy_vds_range_locked(cached, first, count, output);
    }

    template<class RowAt>
    bool copy_vds_rows_if_available_locked(const CachedDataset& cached, std::size_t count,
                                           std::size_t row_bytes, unsigned char* output,
                                           RowAt row_at) const {
        if (!cached.vds) return false;
        if (!ensure_vds_partitions_mapped_locked(cached)) return false;
        for (std::size_t i = 0; i < count; ++i) {
            const auto row = static_cast<std::size_t>(row_at(i));
            const auto& partition = cached.partitions[partition_index(cached, row)];
            const auto local = row - partition.virtual_start;
            const auto* base = static_cast<const unsigned char*>(partition.source->mapping.data) + partition.data_offset;
            std::memcpy(output + i * row_bytes, base + local * partition.row_bytes, partition.row_bytes);
        }
        return true;
    }

    bool ensure_vds_partitions_mapped_locked(const CachedDataset& cached) const {
        for (const auto& partition : cached.partitions) {
            if (partition.source->failed) return false;
            if (!partition.source->mapping.map()) {
                partition.source->failed = true;
                return false;
            }
        }
        return true;
    }

    void advise_vds_partitions_locked(const CachedDataset& cached, int advice) const {
        for (const auto& partition : cached.partitions) partition.source->mapping.advise(advice);
    }

    const unsigned char* mapped_base_locked(CachedDataset& cached) const {
        if (cached.mapped_base_checked) return cached.mapped_base;
        cached.mapped_base_checked = true;
        if (!cached.mapped || !ensure_mapping_locked()) return nullptr;
        const auto rows_count = static_cast<std::uint64_t>(cached.shape.empty() ? 1 : cached.shape[0]);
        const auto byte_count = static_cast<std::uint64_t>(cached.row_bytes);
        const auto offset = static_cast<std::uint64_t>(cached.offset);
        const auto file_size = static_cast<std::uint64_t>(mapping_.size);
        if (rows_count == 0) return nullptr;
        if (byte_count <= std::numeric_limits<std::uint64_t>::max() / rows_count) {
            const auto payload = rows_count * byte_count;
            if (offset <= std::numeric_limits<std::uint64_t>::max() - payload) {
                const auto required = offset + payload;
                if (required <= file_size)
                    cached.mapped_base = static_cast<const unsigned char*>(mapping_.data) + cached.offset;
            }
        }
        return cached.mapped_base;
    }

    template<class RowAt>
    void copy_mapped_rows_locked(const unsigned char* base, std::size_t count,
                                 std::size_t row_bytes, unsigned char* output,
                                 RowAt row_at) const {
        constexpr std::size_t kPrefetchDistance = 3;
        std::size_t i = 0;
        while (i < count) {
            const auto start = static_cast<std::size_t>(row_at(i));
            std::size_t run = 1;
            while (i + run < count && static_cast<std::size_t>(row_at(i + run)) == start + run)
                ++run;
            if (i + kPrefetchDistance < count) {
                const auto next = static_cast<std::size_t>(row_at(i + kPrefetchDistance));
                prefetch_read(base + next * row_bytes);
            }
            std::memcpy(output + i * row_bytes, base + start * row_bytes, run * row_bytes);
            i += run;
        }
    }

    void hdf5_read_rows_locked(const CachedDataset& cached, const Shape& memory_shape,
                               unsigned char* output, const std::string& error) const {
        auto memory = internal::dataspace(memory_shape);
        auto type = internal::datatype(cached.type);
        internal::check(H5Dread(cached.dataset, type, memory, cached.space, H5P_DEFAULT, output), error);
    }

    void select_indexed_rows(hid_t space, const std::vector<std::int64_t>& rows,
                               std::size_t begin, std::size_t end,
                               const Shape& dimensions,
                               const Shape& start, const Shape& stride,
                               const Shape& count, const Shape& block) const {
        // Select small leaf batches directly; larger batches are split so HDF5
        // merges selection trees rather than repeatedly extending one selection.
        // 16 is the empirical crossover point.
        constexpr std::size_t kIndexedSelectionLeafSize = 16;
        if (end - begin < kIndexedSelectionLeafSize) {
            Shape row_start = start;
            Shape row_count = count;
            row_count[0] = 1;
            for (std::size_t i = begin; i < end; ++i) {
                row_start[0] = static_cast<hsize_t>(rows[i]);
                internal::check(
                    H5Sselect_hyperslab(space, H5S_SELECT_OR, row_start.data(), stride.data(),
                                        row_count.data(), block.data()),
                    "Cannot select indexed HDF5 rows.");
            }
            return;
        }
        const auto middle = begin + (end - begin) / 2;
        select_indexed_rows(space, rows, begin, middle, dimensions, start, stride, count, block);
        internal::Handle second(H5Screate_simple(static_cast<int>(dimensions.size()), dimensions.data(), nullptr), H5Sclose);
        internal::check(H5Sselect_none(second), "Cannot initialize indexed selection.");
        select_indexed_rows(second, rows, middle, end, dimensions, start, stride, count, block);
        internal::check(H5Smodify_select(space, H5S_SELECT_OR, second), "Cannot merge indexed HDF5 selections.");
    }

    void read_selected_rows_locked(CachedDataset& cached, const std::vector<std::int64_t>& rows,
                                   const Shape& dimensions, const Shape& memory_shape,
                                   unsigned char* output) const {
        const auto rank = dimensions.size();
        Shape start = filled_shape(rank, 0);
        Shape stride = filled_shape(rank, 1);
        Shape count = dimensions;
        Shape block = filled_shape(rank, 1);
        count[0] = 1;
        auto& space = space_locked(cached);
        internal::check(H5Sselect_none(space), "Cannot initialize indexed selection.");
        if (rows.size() == 1) {
            start[0] = static_cast<hsize_t>(rows[0]);
            internal::check(H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), stride.data(),
                                                count.data(), block.data()),
                            "Cannot select indexed HDF5 rows.");
        } else {
            select_indexed_rows(space, rows, 0, rows.size(), dimensions,
                                start, stride, count, block);
        }
        hdf5_read_rows_locked(cached, memory_shape, output, "Cannot read indexed HDF5 rows.");
    }

    internal::Handle& space_locked(CachedDataset& cached) const {
        if (!cached.space) cached.space = internal::Handle(H5Dget_space(cached.dataset), H5Sclose);
        return cached.space;
    }

    void ensure_vds_prepared_locked(CachedDataset& cached, const std::string& name) const {
        if (cached.vds_checked || sources_.empty() || sources_failed_) return;
        try { prepare_vds_locked(cached, name); }
        catch (const std::bad_alloc&) { throw; }
        catch (...) {}
        cached.vds_checked = true;
    }

    CachedDataset& dataset_locked(const std::string& name) const {
        auto it = cache_.find(name);
        if (it == cache_.end()) {
            CachedDataset entry;
            entry.dataset = internal::dataset(file_, name);
            internal::Handle space(H5Dget_space(entry.dataset), H5Sclose);
            entry.shape = internal::shape(space);
            entry.type = internal::dataset_dtype(entry.dataset);
            const auto dimensions = entry.shape.empty() ? Shape{1} : entry.shape;
            entry.row_bytes = row_bytes_for(dimensions, entry.type);
            if (sec2_driver_ && !entry.shape.empty()) {
                internal::Handle native_type = internal::datatype(entry.type);
                if (const auto offset = contiguous_mapping(entry.dataset, native_type, true)) {
                    entry.mapped = true;
                    entry.offset = *offset;
                }
            }
            it = cache_.emplace(name, std::move(entry)).first;
        }
        return it->second;
    }

    Array read_locked(const std::string& name, Selection selection) const {
        auto& cached = dataset_locked(name);
        ensure_vds_prepared_locked(cached, name);
        const auto& physical = cached.shape;
        const Shape dimensions = physical.empty() ? Shape{1} : physical;
        if (!selection.step) throw std::invalid_argument("slice step cannot be zero");
        if (selection.squeeze && selection.count != 1) throw std::invalid_argument("An integer selection requires exactly one row.");
        if (selection.count) {
            if (selection.start < 0 || static_cast<hsize_t>(selection.start) >= dimensions[0] ||
                selection.step == std::numeric_limits<std::int64_t>::min())
                throw std::out_of_range("HDF5 dataset index is out of range.");
            const auto start_index = static_cast<hsize_t>(selection.start);
            const auto available = selection.step > 0
                ? (dimensions[0] - 1 - start_index) / static_cast<hsize_t>(selection.step)
                : static_cast<hsize_t>(selection.start / -selection.step);
            if (selection.count - 1 > available)
                throw std::out_of_range("HDF5 dataset index is out of range.");
        }
        Shape output = dimensions;
        output[0] = selection.count;
        if (selection.squeeze) output.erase(output.begin());
        Array result(cached.type, output, internal::Initialization::uninitialized);
        if (!result.size()) return result;
        auto* bytes = result.data();
        if (physical.empty()) {
            copy_scalar_rows_locked(cached, selection.count, bytes);
            return result;
        }

        const auto rank = dimensions.size();
        const auto row_bytes = cached.row_bytes;
        const auto row_at = [&](std::size_t index) {
            return static_cast<std::size_t>(
                selection.start + static_cast<std::int64_t>(index) * selection.step);
        };

        if (selection.step == 1) {
            if (copy_vds_range_if_available_locked(cached, static_cast<std::size_t>(selection.start),
                                                   selection.count, bytes))
                return result;
        } else if (copy_vds_rows_if_available_locked(cached, selection.count, row_bytes, bytes, row_at)) {
            return result;
        }

        if (const auto* base = mapped_base_locked(cached)) {
            auto* out = result.data();
            if (selection.step == 1) {
                std::memcpy(out, base + static_cast<std::size_t>(selection.start) * row_bytes,
                            selection.count * row_bytes);
                return result;
            }
            if (selection.step == -1) {
                const auto first = static_cast<std::size_t>(
                    selection.start - static_cast<std::int64_t>(selection.count - 1));
                std::memcpy(out, base + first * row_bytes, selection.count * row_bytes);
                for (std::size_t i = 0; i < selection.count / 2; ++i)
                    std::swap_ranges(out + i * row_bytes, out + (i + 1) * row_bytes,
                                     out + (selection.count - i - 1) * row_bytes);
                return result;
            }
            copy_mapped_rows_locked(base, selection.count, row_bytes, out, row_at);
            return result;
        }

        Shape start = filled_shape(rank, 0);
        Shape stride = filled_shape(rank, 1);
        Shape count = dimensions;
        const auto first_index = selection.step > 0
            ? selection.start
            : selection.start + static_cast<std::int64_t>(selection.count - 1) * selection.step;
        start[0] = static_cast<hsize_t>(first_index);
        stride[0] = static_cast<hsize_t>(selection.step > 0 ? selection.step : -selection.step);
        count[0] = selection.count;
        auto& space = space_locked(cached);
        internal::check(H5Sselect_hyperslab(space, H5S_SELECT_SET, start.data(), stride.data(), count.data(), nullptr), "Cannot select HDF5 rows.");
        hdf5_read_rows_locked(cached, output, bytes, "Cannot read selected HDF5 rows.");
        if (selection.step < 0) {
            for (std::size_t i = 0; i < selection.count / 2; ++i)
                std::swap_ranges(bytes + i * row_bytes, bytes + (i + 1) * row_bytes,
                                 bytes + (selection.count - i - 1) * row_bytes);
        }
        return result;
    }

    void copy_indices_locked(CachedDataset& cached, const Shape& dimensions,
                             const std::vector<std::int64_t>& indices,
                             unsigned char* output) const {
        if (indices.empty()) return;
        if (cached.shape.empty()) {
            copy_scalar_rows_locked(cached, indices.size(), output);
            return;
        }

        const auto row_bytes = cached.row_bytes;
        const auto row_at = [&](std::size_t index) {
            return static_cast<std::size_t>(indices[index]);
        };

        if (copy_vds_rows_if_available_locked(cached, indices.size(), row_bytes, output, row_at))
            return;

        if (const auto* base = mapped_base_locked(cached)) {
            copy_mapped_rows_locked(base, indices.size(), row_bytes, output, row_at);
            return;
        }

        Shape output_shape = dimensions;
        output_shape[0] = indices.size();
        const bool strictly_increasing = std::adjacent_find(
            indices.begin(), indices.end(),
            [](std::int64_t left, std::int64_t right) { return left >= right; }) == indices.end();

        if (strictly_increasing) {
            read_selected_rows_locked(cached, indices, dimensions, output_shape, output);
            return;
        }

        std::vector<std::int64_t> sorted = indices;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

        auto selected_shape = dimensions;
        selected_shape[0] = sorted.size();
        if (row_bytes && sorted.size() > std::numeric_limits<std::size_t>::max() / row_bytes)
            throw std::overflow_error("Array byte size exceeds addressable memory.");
        const auto selected_bytes = sorted.size() * row_bytes;
        std::unique_ptr<unsigned char[]> selected(new unsigned char[selected_bytes]);
        read_selected_rows_locked(cached, sorted, dimensions, selected_shape, selected.get());

        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto offset = static_cast<std::size_t>(
                std::lower_bound(sorted.begin(), sorted.end(), indices[i]) - sorted.begin());
            std::memcpy(output + i * row_bytes, selected.get() + offset * row_bytes, row_bytes);
        }
    }

    Array read_indices_locked(const std::string& name, std::vector<std::int64_t> indices) const {
        auto& cached = dataset_locked(name);
        ensure_vds_prepared_locked(cached, name);
        const auto dimensions = cached.shape.empty() ? Shape{1} : cached.shape;
        normalize_indices(dimensions, indices);
        auto output = dimensions;
        output[0] = indices.size();
        Array result(cached.type, output, internal::Initialization::uninitialized);
        if (!result.size()) return result;
        copy_indices_locked(cached, dimensions, indices, result.data());
        return result;
    }

    NamedArrayViews read_indices_many_locked(const std::vector<std::string>& names,
                                             std::vector<std::int64_t> indices) const {
        cache_.reserve(cache_.size() + names.size());
        std::vector<CachedDataset*> caches;
        caches.reserve(names.size());
        for (const auto& name : names) {
            auto& cached = dataset_locked(name);
            ensure_vds_prepared_locked(cached, name);
            caches.push_back(&cached);
        }

        std::vector<std::size_t> row_counts;
        row_counts.reserve(caches.size());
        for (const auto* cached : caches) {
            const auto dimensions = cached->shape.empty() ? Shape{1} : cached->shape;
            row_counts.push_back(static_cast<std::size_t>(dimensions[0]));
        }

        std::optional<std::vector<std::int64_t>> shared_indices;
        const bool shared_rows = !row_counts.empty() &&
            std::all_of(row_counts.begin(), row_counts.end(),
                        [&](std::size_t rows) { return rows == row_counts.front(); });
        if (shared_rows) {
            shared_indices = indices;
            Shape dimensions{static_cast<hsize_t>(row_counts.front())};
            normalize_indices(dimensions, *shared_indices);
        }

        struct Plan {
            CachedDataset* cached = nullptr;
            Shape dimensions;
            Shape output;
            std::size_t row_bytes = 0;
            std::size_t offset = 0;
            std::vector<std::int64_t> indices;
        };
        std::vector<Plan> plans;
        plans.reserve(caches.size());
        std::size_t total_bytes = 0;
        for (std::size_t i = 0; i < caches.size(); ++i) {
            auto* cached = caches[i];
            const auto dimensions = cached->shape.empty() ? Shape{1} : cached->shape;
            std::vector<std::int64_t> key_indices;
            if (shared_indices) {
                key_indices = *shared_indices;
            } else {
                key_indices = indices;
                normalize_indices(dimensions, key_indices);
            }
            Shape output = dimensions;
            output[0] = key_indices.size();
            const auto row_bytes = cached->row_bytes;
            if (row_bytes && key_indices.size() > std::numeric_limits<std::size_t>::max() / row_bytes)
                throw std::overflow_error("Array byte size exceeds addressable memory.");
            const auto bytes = key_indices.size() * row_bytes;
            const auto alignment = cached->type.size ? cached->type.size : std::size_t{1};
            if (total_bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1))
                throw std::overflow_error("Array byte size exceeds addressable memory.");
            const auto offset = (total_bytes + alignment - 1) & ~(alignment - 1);
            if (offset > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("Array byte size exceeds addressable memory.");
            plans.push_back({cached, dimensions, std::move(output), row_bytes, offset, std::move(key_indices)});
            total_bytes = offset + bytes;
        }

        NamedArrayViews result;
        if (total_bytes) result.storage = std::unique_ptr<unsigned char[]>(new unsigned char[total_bytes]);
        result.views.reserve(plans.size());
        for (std::size_t i = 0; i < plans.size(); ++i) {
            auto& plan = plans[i];
            auto* out = result.storage ? result.storage.get() + plan.offset : nullptr;
            if (out && plan.row_bytes) copy_indices_locked(*plan.cached, plan.dimensions, plan.indices, out);
            result.views.emplace_back(names[i], ArrayView{plan.cached->type, plan.output, out});
        }
        return result;
    }
public:
    explicit Reader(const std::string& path) : file_(internal::open_file(path)) {
        const auto source_paths = internal::sources(file_, path);
        sources_.reserve(source_paths.size());
        for (const auto& source_path : source_paths) {
            auto source = std::make_shared<SourceFile>();
            try {
                source->file = internal::open_file(source_path);
                duplicate_sec2_descriptor(source->file, source->mapping.fd, source->mapping.size);
            } catch (const std::bad_alloc&) {
                throw;
            } catch (...) {
                source->failed = true;
            }
            if (source->mapping.fd < 0 || source->mapping.size == 0) source->failed = true;
            sources_failed_ = sources_failed_ || source->failed;
            sources_.push_back(std::move(source));
        }
        sec2_driver_ = duplicate_sec2_descriptor(file_, mapping_.fd, mapping_.size);
    }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    std::string store() const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return internal::string_attribute(file_, store_attribute).value_or("");
    }
    std::vector<std::string> keys(const std::string& root = "") const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return internal::dataset_paths(file_, root);
    }
    std::vector<std::string> columns(const std::string& name = "primal") const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        if (internal::object_type(file_, name) == H5O_TYPE_DATASET) return {};
        auto parent = internal::group(file_, name);
        return internal::stored_keys(parent);
    }
    std::vector<std::string> row_keys() const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto& cached = dataset_locked(row_store_dataset);
        const auto keys = internal::json_attribute<std::vector<std::string>>(cached.dataset, keys_attribute);
        if (!keys) throw std::invalid_argument("Row store has no dsbo_keys attribute.");
        return *keys;
    }
    Shape shape(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return dataset_locked(name).shape;
    }
    DType type(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return dataset_locked(name).type;
    }
    std::vector<hsize_t> partition_sizes(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return internal::partition_sizes(dataset_locked(name).dataset);
    }
    std::optional<Statistics> statistics(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return internal::read_statistics(dataset_locked(name).dataset);
    }
    Array read(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        auto& cached = dataset_locked(name);
        ensure_vds_prepared_locked(cached, name);
        if (cached.vds && !cached.shape.empty() && cached.shape[0] > 0) {
            const auto row_bytes = cached.row_bytes;
            if (!row_bytes) return Array(cached.type, cached.shape, internal::Initialization::uninitialized);
            if (ensure_vds_partitions_mapped_locked(cached)) {
                Array result(cached.type, cached.shape, internal::Initialization::uninitialized);
                advise_vds_partitions_locked(cached, MADV_SEQUENTIAL);
                const bool copied = copy_vds_range_locked(cached, 0, cached.shape[0], result.data());
                advise_vds_partitions_locked(cached, MADV_NORMAL);
                if (copied) return result;
            }
        }
        if (cached.mapped && !cached.shape.empty() && cached.shape[0] > 0) {
            const auto row_bytes = cached.row_bytes;
            if (!row_bytes) return Array(cached.type, cached.shape, internal::Initialization::uninitialized);
            if (const auto* base = mapped_base_locked(cached)) {
                Array result(cached.type, cached.shape, internal::Initialization::uninitialized);
                mapping_.advise(MADV_SEQUENTIAL);
                std::memcpy(result.data(), base, result.nbytes());
                mapping_.advise(MADV_NORMAL);
                return result;
            }
        }
        return internal::read_dataset(cached.dataset);
    }
    Array read(const std::string& name, Selection selection) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return read_locked(name, selection);
    }
    Array read_indices(const std::string& name, std::vector<std::int64_t> indices) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return read_indices_locked(name, std::move(indices));
    }
    NamedArrayViews read_indices_many(const std::vector<std::string>& names,
                                      std::vector<std::int64_t> indices) const {
        std::lock_guard<std::mutex> lock(mutex_); require_open();
        return read_indices_many_locked(names, std::move(indices));
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        mapping_.reset();
        cache_.clear();
        sources_.clear();
        file_.close_checked();
    }
};

} // namespace dsbo::io
