// What:    C interface for linking to libcukcoo
// Why:     Rust doesn't link directly with C++, currently.
// Desc:    Make one instance of the cuckoo hash table and provide
//          pass-through functions to each of the instance members.
// Use of this interface should only be done via the provided Rust
// wrapper for safety.  This file compiles to a library that
// includes libcityhash.a The safe Rust interface will link against
// this derivative library which exports the below C interface, hiding
// the libcuckoo C++ interface. The file build.rs will handle
// compiling us.
//
// Test this code compiles correctly with 'make'

#include <iostream>
#include <cstdlib>
#include <mutex>
#include <cuckoohash_map.hh>

#define __inline __attribute__((always_inline))

// We have to use custom comparators and hashers because we are using
// C strings (char*) for our keys, and the templated cuckoo table and
// friends treat C strings and memory addresses (they don't look at
// the char buffer)

extern uint64_t CityHash64(const char *buf, size_t len);

// Use cityhasher from google to calculate hashes of C strings.
// Reference: thirdparty/libcukcoo/src/default_hasher.hh
// Hard-coded for char* types
class CStringHasher {
    public:
        // we assume NUL-termination
        size_t __inline operator()(const char *k) const {
            return (size_t)CityHash64(k, strlen(k));
        }
};

class U64Hasher {
    public:
        size_t __inline operator()(uint64_t k) const {
            return (size_t)CityHash64((char*)&k, sizeof(k));
        }
};

// Comparator for C strings that examines the entire buffer.
// Reference: cppreference.com/w/cpp/utility/functional/equal_to
// Hard-coded for char* types
class CStringEqual : public std::equal_to<char*> {
    public:
        // we assume NUL-termination
        bool __inline operator() (const char *lhs,
                const char *rhs) const {
            return (0 == strcmp(lhs, rhs));
        }
};

extern "C" {
    // Value type should be a primitive, as we've coded function
    // parameters to copy-by-value (no references). If that changes,
    // the interface should be made with references (l- or r-value).
    typedef uint64_t KType;
    typedef uint64_t VType;

    // Singleton instance of the hash table.
    static cuckoohash_map<KType, VType,
        U64Hasher> *cuckoomap = nullptr;

    // Lock used only for initialization.
    static std::mutex singleton_lock;

    void libcuckoo_init(size_t numa_mask, size_t nnodes) {
        if (!cuckoomap && singleton_lock.try_lock()) {
            cuckoomap = new cuckoohash_map<KType,VType,
                      U64Hasher>(numa_mask, nnodes);
            if (!cuckoomap) {
                std::cerr << "Error: OOM allocating cuckoo table"
                    << std::endl;
                std::cerr.flush();
                abort();
            }
            singleton_lock.unlock();
        }
        // if multiple threads try to call init, and we did not get
        // the lock, wait for the winning thread to complete
        // initialization
        while (!cuckoomap)
            ;
    }

    void libcuckoo_clear(void) {
        assert(cuckoomap);
        cuckoomap->clear();
    }

    size_t libcuckoo_size(void) {
        assert(cuckoomap);
        return cuckoomap->size();
    }

    bool libcuckoo_empty(void) {
        assert(cuckoomap);
        return cuckoomap->empty();
    }

    bool libcuckoo_find(const KType key, VType &value) {
        assert(cuckoomap);
        return cuckoomap->find(key, value);
    }

    bool libcuckoo_contains(const KType key) {
        assert(cuckoomap);
        bool ret = cuckoomap->contains(key);
        return ret;
    }


    bool libcuckoo_insert(const KType key, VType value) {
        assert(cuckoomap);
        try {
            return cuckoomap->insert(key, value);
        }
        // FIXME do something more meaningful with these
        catch (libcuckoo_load_factor_too_low &e) {
            std::cerr << "Error: cuckoo load factor too low"
                << std::endl;
            std::cerr.flush();
            abort();
        }
        catch (libcuckoo_maximum_hashpower_exceeded &e) {
            std::cerr << "Error: cuckoo load max hashpower exceeded"
                << std::endl;
            std::cerr.flush();
            abort();
        }
        return false; // currently not reached
    }

    bool libcuckoo_erase(const KType key, VType &value) {
        assert(cuckoomap);
        return cuckoomap->erase2(key, &value);
    }

    // return true if value was replaced (old item made
    // available), or false if item was inserted (nothing replaced)
    bool libcuckoo_update(const KType key,
            VType value, VType &old) {
        assert(cuckoomap);
        return cuckoomap->update_insert(key, value, old);
    }

    bool libcuckoo_reserve(size_t n) {
        assert(cuckoomap);
        try {
            return cuckoomap->reserve(n);
        }
        // FIXME do something more meaningful with these
        catch (libcuckoo_maximum_hashpower_exceeded &e) {
            std::cerr << "Error: cuckoo load max hashpower exceeded"
                << std::endl;
            std::cerr.flush();
            abort();
        }
    }

    void libcuckoo_print_conflicts(size_t pct) {
        assert(cuckoomap);
        auto v = cuckoomap->lock_conflicts(pct);
        long buckets[10]; // [0] is 0-10 percent conflicts, etc.
        memset(buckets, 0, 10*sizeof(long));
        for (auto p : v) {
            size_t percent = p.second;
            buckets[ percent/10 ]++;
        }
        std::cout << "Histogram of lock conflicts: [ ";
        for (size_t i = 0; i < 10; i++) {
            std::cout << buckets[i] << ", ";
        }
        std::cout << "]" << std::endl;
    }
}
